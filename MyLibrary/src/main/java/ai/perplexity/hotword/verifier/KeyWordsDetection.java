/* FULL FILE — BULK-HEAD EMBEDDING BATCH + TAIL PER-FRAME — NO API REMOVED */

package ai.perplexity.hotword.verifier;

import ai.onnxruntime.*;
import android.content.Context;
import java.nio.FloatBuffer;
import java.util.*;
import android.util.Log;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.DataOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import android.content.res.AssetManager;
import java.util.function.Function;
import android.media.AudioFormat;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.content.Intent;
import android.os.Build;

import java.util.stream.Collectors;
import android.app.ActivityManager;
import java.util.Date;

import android.os.Environment;
import android.provider.Settings;
import android.net.Uri;

import java.util.function.BiConsumer;
import android.Manifest;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicBoolean;

public class KeyWordsDetection {

    private static final String DM_DELIM = "####%$#$%$#^&^*&*^#$%#$%#$#%^&&*****###";
    private String vadPath = null;

    private OrtEnvironment env = null;
    private OrtSession[] sessions;
    private OrtSession melspecSession;
    private OrtSession embeddingSession;
    private final String TAG = "KeyWordsDetection";
    private Context context;
    private Map<String, OnnxTensor> inputs;
    private List<String>[] inputNames;
    private int[] nFeatureFrames;
    private volatile boolean isListening;
    private int bufferSize = 0;
    private Set<String> embeddingInputNames;
    private Set<String> melspecInputNames;
    private static final int MEL_SPECTROGRAM_MAX_LEN = 10 * 97;
    private static final int SAMPLE_RATE = 16000; // currently
    private static final int RAW_BUFFER_MAX_LEN = Constants.FRAME_LENGTH * 2;
    private final int featureBufferMaxLen = 120;
    private Deque<float[]> featureDeque = new ArrayDeque<>();

    private Deque<Short> rawDataBuffer = new ArrayDeque<>(RAW_BUFFER_MAX_LEN);
    private ArrayList<float[]> melspectrogramBuffer;
//    private String audioWavPath;
    private int[] keyBufferCnts;
    private float[] keyThreasholds;
    private float[] fakeThresholds;
    private BiConsumer<Boolean, String> keywordDetectedCallback;
    private int[] concurrentPredictions;
    private long[] lastCallbackInMS;

    private int bufferPosition = 0;
    private final short[] audioBuffer;
    private static final int LAST_SEC_BUFF_SIZE = SAMPLE_RATE * 3; // 3 seconds buffer
    private String[] strippedModelNames;
    private String[] internalModelPaths;
    private String lastFile = "";
    private long[] msBetweenCallbacks;
    private boolean isSticky = true;
    private static int randomDataSize = 16000 * 4;
    private short[] randomData = new short[randomDataSize];

    private Thread keyWordDetectionThread = null;
    private Thread VADThread = null;

    // VAD API:
    private volatile boolean isVADListening = false;
    private float VADThreshold = 0.45f;
    private int msWindow = 1000;

    // --- External push mode (no mic thread) ---
    private final AtomicBoolean isExternalMode = new AtomicBoolean(false);
    private short[] extCarry = new short[Constants.FRAME_LENGTH]; // leftover < 1280 from prior push
    private int extCarryLen = 0;

    // Bulk optimize controls
    private volatile boolean bulkOptimizeEnabled = true;
    private volatile float   bulkOptimizeRatio   = 0.80f; // start predicting after this ratio of audio has been ingested
    private volatile int     bulkMinSamples      = 5000;

    // === Bulk-embedding (head) collection ===
    private final ArrayList<float[][]> bulkWindows = new ArrayList<>(512);
    private boolean bulkCollecting = false; // true while ingesting the "head" of a bulk push

    // Reusable tiny objects to reduce churn
    private final Map<String, OnnxTensor> embeddingInputsReusable = new HashMap<>();

    // === Timing helpers (added) ===
    private static long tNow() { return System.nanoTime(); }
    private static String tMs(long t0) {
        return String.format(Locale.US, "%.2f ms", (System.nanoTime() - t0) / 1_000_000.0);
    }
    private static final boolean DEBUG = false;
    
    static {
        try {
            System.loadLibrary("onnxruntime");
            Log.d("KeyWordsDetection", "onnxruntime loaded successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.w("KeyWordsDetection", "Native library not found or symbol missing: " + e.getMessage(), e);
        } catch (SecurityException e) {
            Log.w("KeyWordsDetection", "Security violation while loading native library: " + e.getMessage(), e);
        } catch (Exception e) {
            Log.w("KeyWordsDetection", "Unexpected error while loading native libraries: " + e.getMessage(), e);
        } catch (Throwable t) {
            Log.w("KeyWordsDetection", "Critical error while loading native libraries", t);
        }
        try {
            System.loadLibrary("onnxruntime4j_jni");
            Log.d("KeyWordsDetection", "onnxruntime4j_jni loaded successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.w("KeyWordsDetection", "Native library not found or symbol missing: " + e.getMessage(), e);
        } catch (SecurityException e) {
            Log.w("KeyWordsDetection", "Security violation while loading native library: " + e.getMessage(), e);
        } catch (Exception e) {
            Log.w("KeyWordsDetection", "Unexpected error while loading native libraries: " + e.getMessage(), e);
        } catch (Throwable t) {
            Log.w("KeyWordsDetection", "Critical error while loading native libraries", t);
        }
    }

    // New constructor with fastSession flag
    public KeyWordsDetection(Context context,
                            String[] modelPaths,
                            float[] thresholds,
                            int[] bufferCnts,
                            long[] msBetweenCallback)
            throws OrtException, SecurityException {

        Log.d(TAG, "KeyWordsDetection new constructor: ");
        this.keyThreasholds = thresholds;
        this.keyBufferCnts = bufferCnts;
        this.msBetweenCallbacks = msBetweenCallback;

        int numModels = modelPaths.length;
        fakeThresholds = new float[numModels];
        lastCallbackInMS = new long[numModels];
        concurrentPredictions = new int[numModels];
        for (int i = 0; i < numModels; i++) {
            fakeThresholds[i] = thresholds[i] - 0.1f;
            lastCallbackInMS[i] = 0;
            concurrentPredictions[i] = 0;
        }

        Log.d(TAG, "KeyWordsDetection constructor: keyThreasholds: " + keyThreasholds);
        Log.d(TAG, "KeyWordsDetection constructor: fakeThresholds: " + fakeThresholds);
        audioBuffer = new short[LAST_SEC_BUFF_SIZE];
        Arrays.fill(audioBuffer, (short) 0);

        Log.d(TAG, "KeyWordsDetection constructor: keyBufferCnts: " + keyBufferCnts);
        inputs = new HashMap<>();

        this.context = context;
        strippedModelNames = new String[modelPaths.length];
        internalModelPaths = new String[modelPaths.length];

        Boolean isPlex = false;

        for (int i = 0; i < modelPaths.length; i++) {
            String name = modelPaths[i];
            name = name.substring(name.lastIndexOf('/') + 1);
            name = name.replaceFirst("[.][^.]+$", "");
            strippedModelNames[i] = name;

            String local = assetExists(modelPaths[i])
                 ? copyAssetToInternalStorage(modelPaths[i])
                 : modelPaths[i];
            internalModelPaths[i] = expandDmReturnFirstOnnxIfNeeded(local);
        }

        String[] layer1 = extractLayer1IfPresent();
        String melspecPath, embeddingPath;
        if (layer1 != null) {
            melspecPath   = layer1[0];
            embeddingPath = layer1[1];
            vadPath = layer1[2];
            Log.i(TAG, "Loaded Layer1 from layer1.dm");
        } else {
            melspecPath   = copyAssetIfExists("melspectrogram.onnx");
            embeddingPath = copyAssetIfExists("embedding_model.onnx");
            vadPath = copyAssetIfExists("silero_vad.onnx");
        }
        Log.i(TAG, " layer1[0]:" + melspecPath + "layer1[1]" +  embeddingPath + "layer1[2]" + vadPath);

        Log.i(TAG, "internalModelPaths[0] == " + internalModelPaths[0]);
        if (internalModelPaths[0] == null || melspecPath == null || embeddingPath == null) {
            Log.e(TAG, "KeyWordsDetection: Model path is null after copying asset.");
            throw new OrtException("Failed to copy model asset to internal storage.");
        }

        try {
            Log.d(TAG, "KeyWordsDetection constructor: Models: " + Arrays.toString(modelPaths));
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();

            final int cores = Math.max(1, Runtime.getRuntime().availableProcessors());

            // boolean nnapiAdded = false;
            // try {
            //     // Try NNAPI (may or may not help depending on model ops/quant)
            //     options.addNnapi();
            //     nnapiAdded = true;
            //     Log.i(TAG, "NNAPI EP added successfully.");
            // } catch (Exception e) {
            //     Log.w(TAG, "NNAPI EP not available or failed to load.", e);
            // }

            try {
                Map<String, String> xnnpackOpts = Collections.emptyMap();
                options.addXnnpack(xnnpackOpts);
                Log.i(TAG, "XNNPACK EP added successfully.");
            } catch (Exception e) {
                Log.w(TAG, "XNNPACK EP not available or failed to load.", e);
            }

            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.min(4, cores));
            options.setInterOpNumThreads(1);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
            options.setCPUArenaAllocator(true);
            options.setMemoryPatternOptimization(true);
            options.setSessionLogVerbosityLevel(0);

            melspecSession = env.createSession(melspecPath, options);
            embeddingSession = env.createSession(embeddingPath, options);

            embeddingInputNames = embeddingSession.getInputNames();
            melspecInputNames = melspecSession.getInputNames();

            rawDataBuffer = new ArrayDeque<>(RAW_BUFFER_MAX_LEN);
            melspectrogramBuffer = new ArrayList<>();
            sessions = new OrtSession[modelPaths.length];
            inputNames      = new ArrayList[modelPaths.length];
            nFeatureFrames  = new    int  [modelPaths.length];
            for (int i = 0; i < modelPaths.length; i++) {
                sessions[i] = env.createSession(internalModelPaths[i], options);
                inputNames[i] = new ArrayList<>(sessions[i].getInputNames());
                Map<String, NodeInfo> inputInfoMap = sessions[i].getInputInfo();
                NodeInfo inputInfo = inputInfoMap.get(inputNames[i].get(0));
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();
                nFeatureFrames[i] = (int) shape[1];
                Log.d(TAG, "Model [" + strippedModelNames[i] + "] nFeatureFrames: " + nFeatureFrames[i]);
            }
            if (melspectrogramBuffer != null)
                melspectrogramBuffer.clear();
            else
                melspectrogramBuffer = new ArrayList<>();

            for (int i = 0; i < 76; i++) {
                float[] row = new float[32];
                for (int j = 0; j < 32; j++) {
                    row[j] = 1.0f;
                }
                melspectrogramBuffer.add(row);
            }
            printMelspectrogramBuffer();

            if (featureDeque != null)
                featureDeque.clear();
            else
                featureDeque = new ArrayDeque<>();

            Random rand = new Random();
            for (int i = 0; i < randomDataSize; i++) {
                randomData[i] = (short) (rand.nextInt(2000) - 1000);
            }
            featureDeque.clear();
            float[][] randomEmbeds = getEmbeddings(randomData);
            if (randomEmbeds != null) {
                for (float[] row : randomEmbeds) {
                    updateFeatureQueue(row);
                }
            }
        } catch (OrtException e) {
            e.printStackTrace();
            throw e;
        }
    }


    public void close() {
        Log.d(TAG, "KeyWordsDetection close() called, cleaning up all native resources.");

        stopListening();

        if (sessions != null) {
            for (int i = 0; i < sessions.length; i++) {
                if (sessions[i] != null) {
                    try {
                        sessions[i].close();
                    } catch (Exception e) {
                        Log.w(TAG, "Failed to close session[" + i + "]: " + e.getMessage());
                    }
                    sessions[i] = null;
                }
            }
        }
        if (melspecSession != null) {
            try { melspecSession.close(); } catch (Exception ignored) {}
            melspecSession = null;
        }
        if (embeddingSession != null) {
            try { embeddingSession.close(); } catch (Exception ignored) {}
            embeddingSession = null;
        }
        if (env != null) {
            try { env.close(); } catch (Exception ignored) {}
            env = null;
        }

        keyWordDetectionThread = null;
        VADThread = null;

        keywordDetectedCallback = null;
        featureDeque.clear();
        melspectrogramBuffer.clear();
        rawDataBuffer = null;

        isListening = false;
        Log.d(TAG, "KeyWordsDetection close() completed.");
    }

    public String getKeywordDetectionModel() {
        if (strippedModelNames != null)
            return strippedModelNames[0];
        return "";
    }

    public String searchAndCopyFileToExternalStorage(String fileName) {
        String[] possibleDirs = {
            context.getFilesDir().getAbsolutePath(),
            "/storage/emulated/0/",
            "/data/data/com.exampleapp/files/",
        };

        for (String dir : possibleDirs) {
            File file = new File(dir, fileName);
            if (file.exists()) {
                Log.d("File Search", "File found: " + file.getAbsolutePath());
                return copyFileToExternalStorage(file.getAbsolutePath());
            } else {
                Log.d("File Search", "File not found in: " + dir);
            }
        }

        Log.e("File Search", "File does not exist in any of the searched directories.");
        return null;
    }

    private String copyFileToExternalStorage(String sourcePath) {
        File sourceFile = new File(sourcePath);
        if (!sourceFile.exists()) {
            Log.e("copyFileToExternal", "File does not exist: " + sourcePath);
            return null;
        }

        File externalDir = new File(context.getExternalFilesDir(null), "MyAppFiles");
        if (!externalDir.exists()) {
            if (!externalDir.mkdirs()) {
                Log.e("copyFileToExternal", "Failed to create directory: " + externalDir.getAbsolutePath());
                return null;
            }
        }

        File destFile = new File(externalDir, sourceFile.getName());

        if (destFile.exists()) {
            boolean deleted = destFile.delete();
            if (!deleted) {
                Log.e("copyFileToExternal", "Failed to delete existing file: " + destFile.getAbsolutePath());
                return null;
            }
        }

        try {
            FileInputStream in = new FileInputStream(sourceFile);
            FileOutputStream out = new FileOutputStream(destFile);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            in.close();
            out.close();
            Log.d("copyFileToExternal", "File copied to: " + destFile.getAbsolutePath());
            return destFile.getAbsolutePath();
        } catch (IOException e) {
            Log.e("copyFileToExternal", "Error copying file: " + e.getMessage());
            return null;
        }
    }

    private boolean isServiceRunning(Class<?> serviceClass) {
        ActivityManager manager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        for (ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
            if (serviceClass.getName().equals(service.service.getClassName())) {
                return true;
            }
        }
        return false;
    }

    public String getRecordingWav() {
        return searchAndCopyFileToExternalStorage(lastFile);
    }

    public void replaceKeywordDetectionModel(Context context, String modelPath, float threshold, int buffer_cnt) throws OrtException {
        replaceKeywordDetectionModel(
                    context,
                new String[]{modelPath},
                new float[]{threshold},
                new int[]{buffer_cnt},
                new long[]{2000L}
                );
    }

    public void replaceKeywordDetectionModel(Context context, String[] modelPaths, float[] thresholds, int[] bufferCnts, long[] msBetweenCallback)
        throws OrtException, SecurityException {
        Log.d(TAG, "replaceKeywordDetectionModel()");
        this.keyThreasholds = thresholds;
        this.keyBufferCnts = bufferCnts;
        this.msBetweenCallbacks = msBetweenCallback;

        // === Model name validation ===
        for (String path : modelPaths) {
            String name = path.substring(path.lastIndexOf('/') + 1).toLowerCase(Locale.ROOT);
            if (!(name.contains("hey_plex") || name.contains("hey_perplexity"))) {
                Log.e(TAG, "Unauthorized usage of DaVoice: " + name);
                throw new SecurityException("Unauthorized usage of DaVoice license! please contact info@davoice.io");
            }
        }

        int numModels = modelPaths.length;
        fakeThresholds = new float[numModels];
        lastCallbackInMS = new long[numModels];
        concurrentPredictions = new int[numModels];
        for (int i = 0; i < numModels; i++) {
            fakeThresholds[i] = thresholds[i] - 0.1f;
            lastCallbackInMS[i] = 0;
            concurrentPredictions[i] = 0;
        }
        strippedModelNames = new String[modelPaths.length];
        internalModelPaths = new String[modelPaths.length];

        for (int i = 0; i < modelPaths.length; i++) {
            String name = modelPaths[i];
            name = name.substring(name.lastIndexOf('/') + 1);
            name = name.replaceFirst("[.][^.]+$", "");
            strippedModelNames[i] = name;

            internalModelPaths[i] = copyAssetToInternalStorage(modelPaths[i]);
            if (internalModelPaths[i] == null) {
                Log.e(TAG, "replaceKeywordDetectionModel(): Model path is null after copying asset.");
                throw new OrtException("Failed to copy model asset to internal storage.");
            }
        }

        if (sessions != null) {
            for (int i = 0; i < sessions.length; i++) {
                if (sessions[i] != null) {
                    try {
                        sessions[i].close();
                    } catch (Exception e) {
                        Log.w(TAG, "Failed to close session[" + i + "]: " + e.getMessage());
                    }
                    sessions[i] = null;
                }
            }
        }

        try {
            Log.d(TAG, "KeyWordsDetection constructor: Models: " + Arrays.toString(modelPaths));
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setInterOpNumThreads(1);
            options.setIntraOpNumThreads(1);
            sessions = new OrtSession[modelPaths.length];
            for (int i = 0; i < modelPaths.length; i++) {
                sessions[i] = env.createSession(internalModelPaths[i], options);
                inputNames[i] = new ArrayList<>(sessions[i].getInputNames());
                Map<String, NodeInfo> inputInfoMap = sessions[i].getInputInfo();
                NodeInfo inputInfo = inputInfoMap.get(inputNames[i].get(0));
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();
                nFeatureFrames[i] = (int) shape[1];
                Log.d(TAG, "Model [" + strippedModelNames[i] + "] nFeatureFrames: " + nFeatureFrames[i]);
            }

        } catch (OrtException e) {
            e.printStackTrace();
            throw e;
        }
    }

    public void printMelspectrogramBuffer() {
        int numRows = melspectrogramBuffer.size();
        int numCols = numRows > 0 ? melspectrogramBuffer.get(0).length : 0;
    }

    private String copyAssetToInternalStorage(String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        File parentFile = file.getParentFile();
        if (!parentFile.exists()) {
            boolean dirsCreated = parentFile.mkdirs();
        }

        if (file.exists()) {
            boolean deleted = file.delete();
            Log.d(TAG, "Existing asset deleted: " + deleted);
        }

        if (!file.exists()) {
            try {
                AssetManager assetManager = context.getAssets();
                InputStream in = assetManager.open(assetName);
                FileOutputStream out = new FileOutputStream(file);
                byte[] buffer = new byte[1024];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    out.write(buffer, 0, read);
                }
                in.close();
                out.close();
            } catch (IOException e) {
                Log.e(TAG, "Failed to copy asset file: " + assetName, e);
                return null;
            }
        }
        return file.getAbsolutePath();
    }

    public void initialize(BiConsumer<Boolean, String> callback) {
        this.keywordDetectedCallback = callback;
        if (concurrentPredictions != null)
            for (int i = 0; i < concurrentPredictions.length; i++) {
                concurrentPredictions[i] = 0;
            }
    }

    public void storeFrame(short[] frame, int frameLength) {
        int remainingSpace = LAST_SEC_BUFF_SIZE - bufferPosition;
        if (frameLength > remainingSpace) {
            bufferPosition = 0;
        }
        System.arraycopy(frame, 0, audioBuffer, bufferPosition, frameLength);
        bufferPosition += frameLength;
    }

    public void flushBufferToWav(String fileName) throws IOException {
        File file = new File(context.getFilesDir(), fileName);
        Log.d(TAG, "Saving WAV file: " + fileName);

        try (FileOutputStream fos = new FileOutputStream(file);
             ByteArrayOutputStream baos = new ByteArrayOutputStream();
             DataOutputStream dos = new DataOutputStream(baos)) {

            int sampleRate = SAMPLE_RATE;
            int numChannels = 1;
            int bitsPerSample = 16;
            int byteRate = sampleRate * numChannels * (bitsPerSample / 8);

            dos.writeBytes("RIFF");
            dos.writeInt(Integer.reverseBytes(36 + LAST_SEC_BUFF_SIZE * 2));
            dos.writeBytes("WAVE");
            dos.writeBytes("fmt ");
            dos.writeInt(Integer.reverseBytes(16));
            dos.writeShort(Short.reverseBytes((short) 1));
            dos.writeShort(Short.reverseBytes((short) numChannels));
            dos.writeInt(Integer.reverseBytes(sampleRate));
            dos.writeInt(Integer.reverseBytes(byteRate));
            dos.writeShort(Short.reverseBytes((short) (numChannels * (bitsPerSample / 8))));
            dos.writeShort(Short.reverseBytes((short) bitsPerSample));
            dos.writeBytes("data");
            dos.writeInt(Integer.reverseBytes(LAST_SEC_BUFF_SIZE * 2));

            byte[] byteBuffer = new byte[LAST_SEC_BUFF_SIZE * 2];
            for (int i = 0; i < LAST_SEC_BUFF_SIZE; i++) {
                int bufferIndex = (bufferPosition + i) % LAST_SEC_BUFF_SIZE;
                byteBuffer[i * 2] = (byte) (audioBuffer[bufferIndex] & 0x00FF);
                byteBuffer[i * 2 + 1] = (byte) ((audioBuffer[bufferIndex] >> 8) & 0x00FF);
            }
            dos.write(byteBuffer, 0, LAST_SEC_BUFF_SIZE * 2);

            fos.write(baos.toByteArray());
            lastFile = fileName;
            Log.d(TAG, "lastFile: " + lastFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void ignoreBatteryOptimization() {
        Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:" + context.getPackageName()));
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(intent);
    }


    public void startListeningExternalAudio(float threshold) {
        if (isListening) {
            Log.d(TAG, "Already listening");
            return;
        }

        try {
            isListening = true;
            isExternalMode.set(true);

            if (concurrentPredictions != null) {
                Arrays.fill(concurrentPredictions, 0);
            }

            inputs = new HashMap<>();

            if (rawDataBuffer != null) rawDataBuffer.clear();
            rawDataBuffer = null;
            rawDataBuffer = new ArrayDeque<>(RAW_BUFFER_MAX_LEN);

            Arrays.fill(audioBuffer, (short) 0);
            bufferPosition = 0;

            if (melspectrogramBuffer != null) melspectrogramBuffer.clear();
            else melspectrogramBuffer = new ArrayList<>();
            for (int i = 0; i < 76; i++) {
                float[] row = new float[32];
                Arrays.fill(row, 1.0f);
                melspectrogramBuffer.add(row);
            }
            printMelspectrogramBuffer();

            // bulk head state
            bulkWindows.clear();
            bulkCollecting = false;

            Random rand = new Random();
            for (int i = 0; i < randomDataSize; i++) {
                randomData[i] = (short) (rand.nextInt(2000) - 1000);
            }
            featureDeque.clear();
            float[][] randomEmbeds = getEmbeddings(randomData);
            if (randomEmbeds != null) {
                for (float[] row : randomEmbeds) {
                    updateFeatureQueue(row);
                }
            }

            extCarryLen = 0;

            Log.d(TAG, "startListeningExternalAudio(): init complete");
            Log.d(TAG, "keyThreasholds: " + Arrays.toString(keyThreasholds));
            Log.d(TAG, "fakeThresholds: " + Arrays.toString(fakeThresholds));
        } catch (Throwable fatal) {
            Log.e(TAG, "startListeningExternalAudio failed – shutting detector down", fatal);
            stopListening();
        }
    }

    public void setBulkOptimize(boolean enabled, float ratio01) {
        bulkOptimizeEnabled = enabled;
        if (ratio01 < 0f) ratio01 = 0f;
        if (ratio01 > 0.99f) ratio01 = 0.99f;
        bulkOptimizeRatio = ratio01;
    }
    
    public void setBulkMinSamples(int samples) {
        bulkMinSamples = Math.max(Constants.FRAME_LENGTH * 2, samples);
    }

    public boolean predictFromExternalFullBuffer(short[] pcm, int length) {
        if (!isListening || !isExternalMode.get() || pcm == null || length <= 0) return false;
        boolean detected = false;

        final int F = Constants.FRAME_LENGTH; // 1280
        int offset = 0;

        // Merge any carried partial from previous call
        if (extCarryLen > 0) {
            int need = F - extCarryLen;
            int take = Math.min(need, length);
            System.arraycopy(pcm, 0, extCarry, extCarryLen, take);
            extCarryLen += take;
            offset += take;

            if (extCarryLen == F) {
                try {
                    short[] merged = new short[length - offset + F];
                    System.arraycopy(extCarry, 0, merged, 0, F);
                    System.arraycopy(pcm, offset, merged, F, length - offset);
                    pcm = merged;
                    length = merged.length;
                    offset = 0;
                    extCarryLen = 0;
                } catch (Throwable t) {
                    Log.e(TAG, "Failed to merge extCarry; processing immediately", t);
                    detected = processOneKwFrameNoCB(extCarry, F);

                    extCarryLen = 0;
                    if (detected) {
                        return true;
                    }
                }
            }
        }

        // Decide head (bulk collect) size
        int preSamples = 0;
        if (bulkOptimizeEnabled && length - offset >= Math.max(bulkMinSamples, F * 2)) {
            preSamples = (int)((length - offset) * bulkOptimizeRatio);
            preSamples = (preSamples / F) * F;
            preSamples = Math.min(Math.max(0, preSamples), Math.max(0, (length - offset) - F));
        }
        int headEnd = offset + preSamples;

        // === HEAD: mel + collect windows, no embeddings, no prediction
        if (preSamples > 0) bulkCollecting = true;
        while (offset + F <= headEnd && isListening && isExternalMode.get()) {
            short[] frame = new short[F];
            System.arraycopy(pcm, offset, frame, 0, F);
            try {
                detectHeadCollect(frame);
                storeFrame(frame, F);
            } catch (Throwable e) {
                Log.e(TAG, "head collect failed", e);
            }
            offset += F;
        }

        // === BOUNDARY: single batched embedding call for the head
        if (bulkCollecting) {
            try {
                flushBulkEmbeddings(); // pushes head embeddings into featureDeque
            } catch (Throwable e) {
                Log.e(TAG, "flushBulkEmbeddings failed", e);
            }
            bulkCollecting = false;
        }

        // === TAIL: regular per-frame embedding + prediction
        while (offset + F <= length && isListening && isExternalMode.get()) {
            short[] frame = new short[F];
            System.arraycopy(pcm, offset, frame, 0, F);
            detected = processOneKwFrameNoCB(frame, F);
            if (detected) {
                extCarryLen = 0;
                return true;
            }

            offset += F;
        }

        // carry remainder
        int remain = length - offset;
        if (remain > 0) {
            if (extCarry.length < F) extCarry = new short[F];
            System.arraycopy(pcm, offset, extCarry, 0, remain);
            extCarryLen = remain;
        }
        return false;
    }

    public void pushNextFrame(short[] pcm, int length) {
        if (!isListening || !isExternalMode.get() || pcm == null || length <= 0) return;

        final int F = Constants.FRAME_LENGTH; // 1280
        int offset = 0;

        // Merge any carried partial from previous call
        if (extCarryLen > 0) {
            int need = F - extCarryLen;
            int take = Math.min(need, length);
            System.arraycopy(pcm, 0, extCarry, extCarryLen, take);
            extCarryLen += take;
            offset += take;

            if (extCarryLen == F) {
                try {
                    short[] merged = new short[length - offset + F];
                    System.arraycopy(extCarry, 0, merged, 0, F);
                    System.arraycopy(pcm, offset, merged, F, length - offset);
                    pcm = merged;
                    length = merged.length;
                    offset = 0;
                    extCarryLen = 0;
                } catch (Throwable t) {
                    Log.e(TAG, "Failed to merge extCarry; processing immediately", t);
                    processOneKwFrame(extCarry, F, /*doPredict=*/true);
                    extCarryLen = 0;
                }
            }
        }

        // Decide head (bulk collect) size
        int preSamples = 0;
        if (bulkOptimizeEnabled && length - offset >= Math.max(bulkMinSamples, F * 2)) {
            preSamples = (int)((length - offset) * bulkOptimizeRatio);
            preSamples = (preSamples / F) * F;
            preSamples = Math.min(Math.max(0, preSamples), Math.max(0, (length - offset) - F));
        }
        int headEnd = offset + preSamples;

        // === HEAD: mel + collect windows, no embeddings, no prediction
        if (preSamples > 0) bulkCollecting = true;
        while (offset + F <= headEnd && isListening && isExternalMode.get()) {
            short[] frame = new short[F];
            System.arraycopy(pcm, offset, frame, 0, F);
            try {
                detectHeadCollect(frame);
                storeFrame(frame, F);
            } catch (Throwable e) {
                Log.e(TAG, "head collect failed", e);
            }
            offset += F;
        }

        // === BOUNDARY: single batched embedding call for the head
        if (bulkCollecting) {
            try {
                flushBulkEmbeddings(); // pushes head embeddings into featureDeque
            } catch (Throwable e) {
                Log.e(TAG, "flushBulkEmbeddings failed", e);
            }
            bulkCollecting = false;
        }

        // === TAIL: regular per-frame embedding + prediction
        while (offset + F <= length && isListening && isExternalMode.get()) {
            short[] frame = new short[F];
            System.arraycopy(pcm, offset, frame, 0, F);
            processOneKwFrame(frame, F, /*doPredict=*/true);
            offset += F;
        }

        // carry remainder
        int remain = length - offset;
        if (remain > 0) {
            if (extCarry.length < F) extCarry = new short[F];
            System.arraycopy(pcm, offset, extCarry, 0, remain);
            extCarryLen = remain;
        }
    }

    private boolean processOneKwFrameNoCB(short[] frame, int frameLength) {
        boolean doPredict = true;
        try {
            // long t0 = tNow();
            detectFromMicrophone(frame, /*computeEmbedding=*/doPredict, /*flushEmbedNow=*/doPredict);
            storeFrame(frame, frameLength);
            // Log.d(TAG, "processOneKwFrameNoCB (detect+store): " + tMs(t0));

            if (!doPredict) return false;

            for (int i = 0; i < sessions.length; i++) {
                //long tp = tNow();
                float meanPrediction = predictFromBuffer(i);
                //Log.d(TAG, "predict[" + strippedModelNames[i] + "] total: " + tMs(tp) + "  value=" + meanPrediction);

                if (meanPrediction > fakeThresholds[i]) {
                    if (meanPrediction < keyThreasholds[i]) {
                        concurrentPredictions[i] = 0;
                    } else {
                        concurrentPredictions[i]++;
                        if (concurrentPredictions[i] >= keyBufferCnts[i]) {
                            long now = System.currentTimeMillis();
                            if (lastCallbackInMS[i] + msBetweenCallbacks[i] <= now) {
                                lastCallbackInMS[i] = now;
                                String fileName = strippedModelNames[i] + "_prediction.wav";
                                flushBufferToWav(fileName);
                                if (keywordDetectedCallback != null) {
                                    keywordDetectedCallback.accept(true, strippedModelNames[i]);
                                }
                                return true;
                            }
                            concurrentPredictions[i] = 0;
                        }
                    }
                } else {
                    concurrentPredictions[i] = 0;
                }
            }
        } catch (Throwable fatal) {
            Log.e(TAG, "processOneKwFrameNoCB (doPredict) failed – stopping detector", fatal);
            stopListening();
            return false;
        }
        return false;
    }

    private void processOneKwFrame(short[] frame, int frameLength) {
        processOneKwFrame(frame, frameLength, /*doPredict=*/true);
    }

    private void processOneKwFrame(short[] frame, int frameLength, boolean doPredict) {
        try {
            // long t0 = tNow();
            detectFromMicrophone(frame, /*computeEmbedding=*/doPredict, /*flushEmbedNow=*/doPredict);
            storeFrame(frame, frameLength);
            // Log.d(TAG, "processOneKwFrame (detect+store): " + tMs(t0));

            if (!doPredict) return;

            for (int i = 0; i < sessions.length; i++) {
                //long tp = tNow();
                float meanPrediction = predictFromBuffer(i);
                //Log.d(TAG, "predict[" + strippedModelNames[i] + "] total: " + tMs(tp) + "  value=" + meanPrediction);

                if (meanPrediction > fakeThresholds[i]) {
                    if (meanPrediction < keyThreasholds[i]) {
                        concurrentPredictions[i] = 0;
                    } else {
                        concurrentPredictions[i]++;
                        if (concurrentPredictions[i] >= keyBufferCnts[i]) {
                            long now = System.currentTimeMillis();
                            if (lastCallbackInMS[i] + msBetweenCallbacks[i] <= now) {
                                lastCallbackInMS[i] = now;
                                String fileName = strippedModelNames[i] + "_prediction.wav";
                                flushBufferToWav(fileName);
                                if (keywordDetectedCallback != null) {
                                    keywordDetectedCallback.accept(true, strippedModelNames[i]);
                                }
                            }
                            concurrentPredictions[i] = 0;
                        }
                    }
                } else {
                    concurrentPredictions[i] = 0;
                }
            }
        } catch (Throwable fatal) {
            Log.e(TAG, "processOneKwFrame(doPredict) failed – stopping detector", fatal);
            stopListening();
        }
    }

    public boolean debug = false;

    public void startListening(float threshold) throws OrtException {
        if (debug) {
            testTwoRawAssets();
            return;
        }

        if (isListening) {
            Log.d(TAG, "Already listening");
            return;
        }
        Log.e(TAG, "Start Listening with Mic not implemented!!!");
    }

    public void stopListening() {
        if (!isListening) {
            Log.d(TAG, "Stop listening is called while not listening.");
            return;
        }
        isListening = false;

        // Best-effort: flush any pending bulk windows before stopping
        try {
            if (!bulkWindows.isEmpty()) {
                flushBulkEmbeddings();
                bulkWindows.clear();
            }
        } catch (Throwable ignore) {}

        Log.d(TAG, "Stopping to listen");

        if (keyWordDetectionThread != null) {
            try {
                keyWordDetectionThread.join();
                keyWordDetectionThread = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "Error stopping audio thread", e);
            }
        }
        isExternalMode.set(false);
        extCarryLen = 0;

        isListening = false;
    }

    // ===== Bulk head helpers =====

    private void detectHeadCollect(short[] audioData) throws OrtException {
        bufferRawData(audioData);
        streamingMelspectrogram(audioData.length); // updates mel buffer
        collectBulkWindowFromLatestMel();          // queue 76x32 into bulkWindows
    }

    private void collectBulkWindowFromLatestMel() {
        int ndx = getMelNumberOfRows();
        if (ndx < 76) return;
        float[][][][] melspectrogramSlice = getMelspecSubArray(ndx - 76, ndx);
        float[][] flattenedSubArray = flattenSubArray(melspectrogramSlice); // 76 x 32
        bulkWindows.add(flattenedSubArray);
    }

    private void flushBulkEmbeddings() throws OrtException {
        if (bulkWindows.isEmpty()) return;
        runEmbeddingBatch(bulkWindows); // pushes rows into featureDeque
        bulkWindows.clear();
    }

    private void streamingMelspectrogram(int nSamples) throws OrtException {
        //long t0 = tNow();
        if (rawDataBuffer.size() < 400) {
            throw new IllegalArgumentException("The number of input frames must be at least 400 samples @ 16khz (25 ms)!");
        }

        List<Short> list;
        synchronized (rawDataBuffer) {
            list = new ArrayList<>(rawDataBuffer);
        }
        int start = Math.max(0, list.size() - nSamples - 160 * 3);

        int[] intArray = list.subList(start, list.size()).stream()
            .mapToInt(Short::intValue)
            .toArray();

        short[] rawData = new short[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            rawData[i] = (short) intArray[i];
        }

        float[][] melspec = getMelspectrogramShort(rawData);
        updateMelspectrogramBuffer(melspec);
        //Log.d(TAG, "streamingMelspectrogram: " + tMs(t0) + " (nSamples=" + nSamples + ", raw=" + rawData.length + ")");
    }

    public void appendMelspec(float[][] melspec) {
        for (float[] row : melspec) {
            melspectrogramBuffer.add(row);
        }
    }
    public int getMelNumberOfRows() {
        return melspectrogramBuffer.size();
    }

    private void truncateMelspectrogramBuffer() {
        while (melspectrogramBuffer.size() > MEL_SPECTROGRAM_MAX_LEN) {
            melspectrogramBuffer.remove(0);
        }
    }

    private void updateMelspectrogramBuffer(float[][] melspec) {
        appendMelspec(melspec);
        if (getMelNumberOfRows() > MEL_SPECTROGRAM_MAX_LEN) {
            truncateMelspectrogramBuffer();
        }
    }


    private float[] toFloatArray(Deque<Float> deque) {
        float[] array = new float[deque.size()];
        int index = 0;
        for (Float f : deque) {
            array[index++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private synchronized void bufferRawData(short[] x) {
        for (short v : x) {
            if (rawDataBuffer.size() >= RAW_BUFFER_MAX_LEN) {
                rawDataBuffer.pollFirst();
            }
            rawDataBuffer.add(v);
        }
    }

    private void bufferRawDataOld(short[] x) {
        for (short v : x) {
            if (rawDataBuffer.size() >= RAW_BUFFER_MAX_LEN)
                rawDataBuffer.pollFirst();
            rawDataBuffer.add(v);
        }
    }

    private float[][] getMelspectrogramShort(short[] audioData) {
        float[] audioDataFloat = new float[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            audioDataFloat[i] = (float) audioData[i];
        }
        return getMelspectrogram(audioDataFloat);
    }

    public static Object squeeze4D(float[][][][] array) {
        int dim1 = array.length;
        int dim2 = array[0].length;
        int dim3 = array[0][0].length;
        int dim4 = array[0][0][0].length;
        if (dim1 == 1) {
            if (dim2 == 1) {
                if (dim3 == 1) {
                    float[] squeezed = new float[dim4];
                    for (int i = 0; i < dim4; i++) {
                        squeezed[i] = array[0][0][0][i];
                    }
                    return squeezed;
                } else {
                    float[][] squeezed = new float[dim3][dim4];
                    for (int i = 0; i < dim3; i++) {
                        squeezed[i] = array[0][0][i];
                    }
                    return squeezed;
                }
            } else {
                float[][][] squeezed = new float[dim2][dim3][dim4];
                for (int i = 0; i < dim2; i++) {
                    squeezed[i] = array[0][i];
                }
                return squeezed;
            }
        } else {
            return array;
        }
    }

    public static float[][] transform(float[][] array, Function<Float, Float> transformFunction) {
        int rows = array.length;
        int cols = array[0].length;
        float[][] transformedArray = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transformedArray[i][j] = transformFunction.apply(array[i][j]);
            }
        }
        return transformedArray;
    }

    private float[][] getMelspectrogram(float[] audioDataFloat) {
        //long t0 = tNow();

        long[] inputShape = new long[]{1, audioDataFloat.length};
        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;
        String melSpecInputName = null;
        Map<String, OnnxTensor> inputs = null;

        try {
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audioDataFloat), inputShape);

            inputs = new HashMap<>();
            melSpecInputName = melspecInputNames.iterator().next();
            inputs.put(melSpecInputName, inputTensor);

            //long tRun0 = tNow();
            result = melspecSession.run(inputs);
            //Log.d(TAG, "melspecSession.run: " + tMs(tRun0));

            Object output = result.get(0).getValue();
            float[][][][] melspectrogram4D = (float[][][][]) output;
            float[][] melspectrogram = (float [][]) squeeze4D(melspectrogram4D);

            //long tTf0 = tNow();
            Function<Float, Float> melspecTransform = x -> x / 10 + 2;
            float [][] spec = transform(melspectrogram, melspecTransform);
            //Log.d(TAG, "melspec transform: " + tMs(tTf0));
            //Log.d(TAG, "getMelspectrogram total: " + tMs(t0) + " (len=" + audioDataFloat.length + ")");
            return spec;
        } catch (Exception e) {
            Log.e(TAG, "Failed to run melspectrogram session: " + e.getMessage());
            return new float[0][0];
        } finally {
            if (inputs != null) inputs.remove(melSpecInputName);
            melSpecInputName = null;
            if (result != null) result.close();
            if (inputTensor != null) inputTensor.close();
        }
    }

    public static List<float[][]> extractWindows(float[][] spec, int windowSize) {
        List<float[][]> windows = new ArrayList<>();

        for (int i = 0; i <= spec.length - windowSize; i += 8) {
            if (spec.length - i >= windowSize) {
                float[][] window = new float[windowSize][];
                for (int j = 0; j < windowSize; j++) {
                    window[j] = spec[i + j];
                }
                windows.add(window);
            }
        }
        return windows;
    }

    public static float[][][][] expandDims(List<float[][]> windows) {
        int numWindows = windows.size();
        int windowSize = windows.get(0).length;
        int featureSize = windows.get(0)[0].length;

        float[][][][] batch = new float[numWindows][windowSize][featureSize][1];

        for (int i = 0; i < numWindows; i++) {
            float[][] window = windows.get(i);
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < featureSize; k++) {
                    batch[i][j][k][0] = window[j][k];
                }
            }
        }

        return batch;
    }

    public static float[][][] squeeze(float[][][][] batch) {
        int numWindows = batch.length;
        int windowSize = batch[0].length;
        int featureSize = batch[0][0].length;

        float[][][] squeezedBatch = new float[numWindows][windowSize][featureSize];

        for (int i = 0; i < numWindows; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < featureSize; k++) {
                    squeezedBatch[i][j][k] = batch[i][j][k][0];
                }
            }
        }

        return squeezedBatch;
    }

    private float[][] getEmbeddings(short[] audioData) throws OrtException {
        //long t0 = tNow();
        float[][] melspectrogram = getMelspectrogramShort(audioData);
        int windowSize = 76;

        List<float[][]> windows = extractWindows(melspectrogram, windowSize);

        float[][][][] batch = expandDims(windows);
        long[] shape = new long[]{windows.size(), 76, 32, 1};
        float[][][] squeezedBatch = squeeze(batch);
        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;

        try {
            //long tMake = tNow();
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatten3D(squeezedBatch)), shape);
            embeddingInputsReusable.clear();
            String embName = embeddingInputNames.iterator().next();
            embeddingInputsReusable.put(embName, inputTensor);
            //Log.d(TAG, "embeddings make tensor (batch): " + tMs(tMake));

            //long tRun = tNow();
            result = embeddingSession.run(embeddingInputsReusable);
            //Log.d(TAG, "embeddingSession.run (batch): " + tMs(tRun));

            float[][][][] outputTensor = (float[][][][]) result.get(0).getValue();

            int dim0 = outputTensor.length;
            int dim1 = outputTensor[0].length;
            int dim2 = outputTensor[0][0].length;
            int dim3 = outputTensor[0][0][0].length;

            float[][] embeddings = new float[dim0][dim1 * dim2 * dim3];
            for (int i = 0; i < dim0; i++) {
                int index = 0;
                for (int j = 0; j < dim1; j++) {
                    for (int k = 0; k < dim2; k++) {
                        for (int l = 0; l < dim3; l++) {
                            embeddings[i][index++] = outputTensor[i][j][k][l];
                        }
                    }
                }
            }
            //Log.d(TAG, "getEmbeddings (batch) total: " + tMs(t0));
            return embeddings;
        } catch (Exception e) {
            Log.e(TAG, "Failed to run prediction session: " + e.getMessage());
            return null;
        } finally {
            embeddingInputsReusable.clear();
            if (result != null) result.close();
            if (inputTensor != null) inputTensor.close();
        }
    }

    public int getNumberOfColumns() {
        return melspectrogramBuffer.isEmpty() ? 0 : melspectrogramBuffer.get(0).length;
    }

    public float[][][][] getMelspecSubArray(int startRow, int endRow) {
        int numRows = endRow - startRow;
        int numCols = getNumberOfColumns();
        float[][][][] subArray = new float[1][numRows][numCols][1];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                subArray[0][i][j][0] = melspectrogramBuffer.get(startRow + i)[j];
            }
        }
        return subArray;
    }
    private float[][] flattenSubArray(float[][][][] subArray) {
        int numRows = subArray[0].length;
        int numCols = subArray[0][0].length;
        float[][] flattenedArray = new float[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                flattenedArray[i][j] = subArray[0][i][j][0];
            }
        }
        return flattenedArray;
    }

    // MAIN detect (with optional embedding + optional immediate flush)
    private float detectFromMicrophone(short[] audioData, boolean computeEmbedding, boolean flushEmbedNow) throws OrtException {
        //long t0 = tNow();

        bufferRawData(audioData);

        //long tMel = tNow();
        streamingMelspectrogram(audioData.length);
        //Log.d(TAG, "streamingMelspectrogram (detectFromMicrophone): " + tMs(tMel));

        if (!computeEmbedding) {
            //Log.d(TAG, "detectFromMicrophone: skip embedding (head)");
            //Log.d(TAG, "detectFromMicrophone total: " + tMs(t0));
            return 0.0f;
        }

        int ndx = getMelNumberOfRows();
        if (ndx < 76) {
            //Log.d(TAG, "detectFromMicrophone: not enough mel rows yet");
            //Log.d(TAG, "detectFromMicrophone total: " + tMs(t0));
            return 0.0f;
        }

        float[][][][] melspectrogramSlice = getMelspecSubArray(ndx - 76, ndx);
        float[][] flattenedSubArray = flattenSubArray(melspectrogramSlice);

        // Single-window embedding (tail regular path)
        // We push a batch of 1 immediately.
        List<float[][]> one = new ArrayList<>(1);
        one.add(flattenedSubArray);
        runEmbeddingBatch(one); // this will push into featureDeque

        //Log.d(TAG, "detectFromMicrophone total: " + tMs(t0));
        return 0.0f;
    }

    // Back-compat entry point
    public float detectFromMicrophone(short[] audioData) throws OrtException {
        return detectFromMicrophone(audioData, /*computeEmbedding=*/true, /*flushEmbedNow=*/true);
    }

    // Batched embedding runner; pushes to feature queue
    private void runEmbeddingBatch(List<float[][]> windows) throws OrtException {
        if (windows == null || windows.isEmpty()) return;

        //long tEmb = tNow();

        float[][][][] batch4d = expandDims(windows);
        float[][][] squeezed = squeeze(batch4d);
        long[] shape = new long[]{windows.size(), 76, 32, 1};

        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;
        try {
            //long tMake = tNow();
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatten3D(squeezed)), shape);
            embeddingInputsReusable.clear();
            String embName = embeddingInputNames.iterator().next();
            embeddingInputsReusable.put(embName, inputTensor);
            //Log.d(TAG, "embeddings make tensor (batched N=" + windows.size() + "): " + tMs(tMake));

            //long tRun = tNow();
            result = embeddingSession.run(embeddingInputsReusable);
            //Log.d(TAG, "embeddingSession.run (batch N=" + windows.size() + "): " + tMs(tRun));

            float[][][][] outputTensor = (float[][][][]) result.get(0).getValue();
            int dim0 = outputTensor.length;
            int dim1 = outputTensor[0].length;
            int dim2 = outputTensor[0][0].length;
            int dim3 = outputTensor[0][0][0].length;

            for (int i = 0; i < dim0; i++) {
                float[] row = new float[dim1 * dim2 * dim3];
                int index = 0;
                for (int j = 0; j < dim1; j++)
                    for (int k = 0; k < dim2; k++)
                        for (int l = 0; l < dim3; l++)
                            row[index++] = outputTensor[i][j][k][l];
                updateFeatureQueue(row);
            }
            //Log.d(TAG, "getEmbeddingsFromMelspectrogram (batched " + windows.size() + ") total: " + tMs(tEmb));
        } catch (Exception e) {
            Log.e(TAG, "runEmbeddingBatch failed: " + e.getMessage());
        } finally {
            embeddingInputsReusable.clear();
            if (result != null) result.close();
            if (inputTensor != null) inputTensor.close();
        }
    }

    private void updateFeatureQueue(float[] newRow) {
        if (featureDeque.size() >= featureBufferMaxLen) {
            featureDeque.removeFirst();
        }
        featureDeque.addLast(newRow);
    }

    private void updateFeatureQueue(float[][] embedding) {
        if (embedding == null || embedding.length == 0) {
            return;
        }
        float[] newRow = embedding[0];
        if (featureDeque.size() >= featureBufferMaxLen) {
            featureDeque.removeFirst();
        }
        featureDeque.addLast(newRow);
    }

    private float[] flatten3D(float[][][] array) {
        int size = 0;
        for (float[][] subArray : array) {
            for (float[] subSubArray : subArray) {
                size += subSubArray.length;
            }
        }
        float[] flattened = new float[size];
        int index = 0;
        for (float[][] subArray : array) {
            for (float[] subSubArray : subArray) {
                System.arraycopy(subSubArray, 0, flattened, index, subSubArray.length);
                index += subSubArray.length;
            }
        }
        return flattened;
    }

    private float predictFromBuffer(int model_i) {
        //long t0 = tNow();
        if (featureDeque.size() < nFeatureFrames[model_i]) {
            return 0.0f;
        }

        //long tPack = tNow();
        float[][] features = new float[nFeatureFrames[model_i]][96];

        float[][] buf = featureDeque.toArray(new float[featureDeque.size()][]);
        System.arraycopy(buf, buf.length - nFeatureFrames[model_i],
            features, 0, nFeatureFrames[model_i]);

        float[] flattenedFeatures = flatten(features);
        //Log.d(TAG, "predict pack/flatten: " + tMs(tPack));

        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;
        Map<String, OnnxTensor> inputs = null;
        try {
            //long tMake = tNow();
            long[] shape = new long[]{1, nFeatureFrames[model_i], 96};
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flattenedFeatures), shape);
            inputs = new HashMap<>();
            inputs.put(inputNames[model_i].get(0), inputTensor);
            //Log.d(TAG, "predict make tensor: " + tMs(tMake));

            //long tRun = tNow();
            result = sessions[model_i].run(inputs);
            //Log.d(TAG, "predict run[" + strippedModelNames[model_i] + "]: " + tMs(tRun));

            float[][] outputData2D = (float[][]) result.get(0).getValue();
            float prediction = outputData2D[0][0];
            //Log.d(TAG, "predict total: " + tMs(t0));
            return prediction;
        } catch (Exception e) {
            Log.e(TAG, "Failed to run prediction session: " + e.getMessage());
            return 0.0f;
        } finally {
            features = null;
            if (inputs != null)
                inputs.remove(inputNames[model_i].get(0));
            inputs = null;
            flattenedFeatures = null;
            if (result != null) result.close();
            if (inputTensor != null) inputTensor.close();
        }
    }

    private float[] flatten(float[][] array) {
        int size = 0;
        for (float[] subArray : array) {
            size += subArray.length;
        }
        float[] flattened = new float[size];
        int index = 0;
        for (float[] subArray : array) {
            System.arraycopy(subArray, 0, flattened, index, subArray.length);
            index += subArray.length;
        }
        return flattened;
    }

    private float[][] getEmbeddingsFromMelspectrogram(float[][] melspectrogram) throws OrtException {
        //long t0 = tNow();
        long[] shape = new long[]{1, 76, 32, 1};
        String embInputName = null;
        OrtSession.Result result = null;
        OnnxTensor inputTensor = null;
        Map<String, OnnxTensor> inputs = null;
        try {
            //long tFlat = tNow();
            float[] flattenedMelspectrogram = flatten4D(new float[][][][]{new float[][][]{melspectrogram}});
            //Log.d(TAG, "embed flatten: " + tMs(tFlat));

            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flattenedMelspectrogram), shape);
            inputs = new HashMap<>();
            embInputName = embeddingInputNames.iterator().next();
            inputs.put(embInputName, inputTensor);

            //long tRun = tNow();
            result = embeddingSession.run(inputs);
            //Log.d(TAG, "embeddingSession.run: " + tMs(tRun));

            float[][][][] outputTensor = (float[][][][]) result.get(0).getValue();

            int dim0 = outputTensor.length;
            int dim1 = outputTensor[0].length;
            int dim2 = outputTensor[0][0].length;
            int dim3 = outputTensor[0][0][0].length;

            float[][] embeddings = new float[dim0][dim1 * dim2 * dim3];
            for (int i = 0; i < dim0; i++) {
                int index = 0;
                for (int j = 0; j < dim1; j++) {
                    for (int k = 0; k < dim2; k++) {
                        for (int l = 0; l < dim3; l++) {
                            embeddings[i][index++] = outputTensor[i][j][k][l];
                        }
                    }
                }
            }
            //Log.d(TAG, "getEmbeddingsFromMelspectrogram total: " + tMs(t0));
            return embeddings;
        } catch (Exception e) {
            Log.e(TAG, "getEmbeddingsFromMelspectrogram failed: " + e.getMessage());
            return null;
        } finally {
            if (inputs != null) inputs.remove(embInputName);
            embInputName = null;
            inputs = null;
            if (result != null) result.close();
            if (inputTensor != null) inputTensor.close();
        }

    }

    private float[] flatten4D(float[][][][] array) {
        int size = 0;
        for (float[][][] subArray : array) {
            for (float[][] subSubArray : subArray) {
                for (float[] subSubSubArray : subSubArray) {
                    size += subSubSubArray.length;
                }
            }
        }
        float[] flattened = new float[size];
        int index = 0;
        for (float[][][] subArray : array) {
            for (float[][] subSubArray : subArray) {
                for (float[] subSubSubArray : subSubArray) {
                    System.arraycopy(subSubSubArray, 0, flattened, index, subSubSubArray.length);
                    index += subSubSubArray.length;
                }
            }
        }
        return flattened;
    }

    private boolean assetExists(String assetName) {
        try (InputStream is = context.getAssets().open(assetName)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private String copyAssetIfExists(String assetName) {
        return assetExists(assetName) ? copyAssetToInternalStorage(assetName) : null;
    }

    private static byte[] readAllBytes(String path) throws IOException {
        try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(path))) {
            byte[] buf = new byte[8192];
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int n;
            while ((n = in.read(buf)) != -1) baos.write(buf, 0, n);
            return baos.toByteArray();
        }
    }

    private static List<int[]> splitByDelimiter(byte[] data, byte[] delim) {
        List<int[]> ranges = new ArrayList<>();
        int start = 0, i = 0;
        while (i <= data.length - delim.length) {
            boolean match = true;
            for (int j = 0; j < delim.length; j++) {
                if (data[i + j] != delim[j]) { match = false; break; }
            }
            if (match) {
                ranges.add(new int[]{start, i});
                i += delim.length;
                start = i;
            } else {
                i++;
            }
        }
        if (start < data.length) ranges.add(new int[]{start, data.length});
        return ranges;
    }

    private static void extractTarFromBuffer(byte[] data, int start, int end, File outDir) throws IOException {
        int pos = start;
        while (pos + 512 <= end) {
            boolean allZero = true;
            for (int k = 0; k < 512; k++) {
                if (data[pos + k] != 0) { allZero = false; break; }
            }
            if (allZero) break;

            String name   = readNullTermString(data, pos + 0,   100);
            long size     = parseOctal(data,       pos + 124,   12);
            int typeflag  = (data[pos + 156] == 0) ? '0' : (data[pos + 156] & 0xFF);
            String prefix = readNullTermString(data, pos + 345, 155);
            String fullName = (prefix != null && !prefix.isEmpty()) ? (prefix + "/" + name) : name;

            pos += 512;

            if (typeflag == '5') {
                File dir = new File(outDir, fullName);
                if (!dir.exists()) dir.mkdirs();
            } else if (typeflag == '0' || typeflag == 0) {
                File out = new File(outDir, fullName);
                File parent = out.getParentFile();
                if (parent != null && !parent.exists()) parent.mkdirs();
                try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(out))) {
                    long remaining = size;
                    byte[] buf = new byte[8192];
                    long copied = 0;
                    while (remaining > 0) {
                        int chunk = (int)Math.min(buf.length, remaining);
                        System.arraycopy(data, pos, buf, 0, chunk);
                        bos.write(buf, 0, chunk);
                        pos += chunk;
                        remaining -= chunk;
                        copied += chunk;
                    }
                }
                int pad = (int)((512 - (size % 512)) % 512);
                pos += pad;
                continue;
            } else {
                int pad = (int)((512 - (size % 512)) % 512);
                pos += size + pad;
            }
        }
    }

    private static String readNullTermString(byte[] b, int off, int len) {
        int end = off + len;
        int i = off;
        while (i < end && b[i] != 0) i++;
        return new String(b, off, i - off, java.nio.charset.StandardCharsets.US_ASCII).trim();
    }

    private static long parseOctal(byte[] b, int off, int len) {
        long val = 0; int end = off + len; int i = off;
        while (i < end && (b[i] == 0x20)) i++;
        for (; i < end; i++) {
            byte c = b[i];
            if (c == 0 || c == 0x20) break;
            val = (val << 3) + (c - '0');
        }
        return val;
    }

    private static File findFirstFileWithExtension(File dir, String ext) {
        if (dir == null || !dir.exists()) return null;
        File[] files = dir.listFiles();
        if (files == null) return null;
        for (File f : files) {
            if (f.isDirectory()) {
                File r = findFirstFileWithExtension(f, ext);
                if (r != null) return r;
            } else if (f.getName().toLowerCase(Locale.ROOT).endsWith(ext)) {
                return f;
            }
        }
        return null;
    }

    private static File findFileByExactName(File dir, String exact) {
        if (dir == null || !dir.exists()) return null;
        File[] files = dir.listFiles();
        if (files == null) return null;
        for (File f : files) {
            if (f.isDirectory()) {
                File r = findFileByExactName(f, exact);
                if (r != null) return r;
            } else if (f.getName().equals(exact)) {
                return f;
            }
        }
        return null;
    }

    private String expandDmReturnFirstOnnxIfNeeded(String packedPath) {
        if (packedPath == null) return null;
        if (packedPath.endsWith(".onnx")) return packedPath;

        File dmFile = new File(packedPath);
        if (!dmFile.exists()) return packedPath;

        File outDir = new File(context.getFilesDir(),
                "dm_unpack_" + dmFile.getName().replaceAll("[^A-Za-z0-9._-]", "_"));
        if (!outDir.exists() && !outDir.mkdirs()) return packedPath;

        try {
            byte[] all = readAllBytes(packedPath);
            byte[] delim = DM_DELIM.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            List<int[]> ranges = splitByDelimiter(all, delim);
            for (int[] r : ranges) {
                extractTarFromBuffer(all, r[0], r[1], outDir);
            }
            File onnx = findFirstFileWithExtension(outDir, ".onnx");
            return (onnx != null && onnx.exists()) ? onnx.getAbsolutePath() : packedPath;
        } catch (Exception e) {
            return packedPath;
        }
    }

    private Map<String, String> expandDmAndPick(String dmPath, String... exactNames) {
        Map<String, String> out = new HashMap<>();
        if (dmPath == null) return out;
        File dmFile = new File(dmPath);
        if (!dmFile.exists()) return out;

        File outDir = new File(context.getFilesDir(),
                "dm_unpack_" + dmFile.getName().replaceAll("[^A-Za-z0-9._-]", "_"));
        if (!outDir.exists()) outDir.mkdirs();

        try {
            byte[] all = readAllBytes(dmPath);
            byte[] delim = DM_DELIM.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            List<int[]> ranges = splitByDelimiter(all, delim);
            for (int[] r : ranges) extractTarFromBuffer(all, r[0], r[1], outDir);

            for (String name : exactNames) {
                File f = findFileByExactName(outDir, name);
                if (f != null && f.exists()) out.put(name, f.getAbsolutePath());
            }
            return out;
        } catch (Exception e) {
            return out;
        }
    }

    private String[] extractLayer1IfPresent() {
        final String dmAsset = "layer1.dm";
        if (!assetExists(dmAsset)) return null;

        String dmPath = copyAssetToInternalStorage(dmAsset);
        Map<String, String> pick = expandDmAndPick(dmPath,
                "melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx");

        String m = pick.get("melspectrogram.onnx");
        String e = pick.get("embedding_model.onnx");
        String v = pick.get("silero_vad.onnx");

        if (m != null && e != null && v != null) {
            return new String[]{m, e, v};
        }
        return null;
    }

    public Map<String, Object> getVoiceProps() {
        Map<String, Object> result = new HashMap<>();
        result.put("error", "No Error");
        result.put("voiceProbability", 0.0);
        result.put("lastTimeHumanVoiceHeard", 0.0);
        return result;
    }

    public void printAllFilesInInternalStorage() {
        printAllFiles(context.getFilesDir());
        printAllAssets("");
    }

    private void printAllFiles(File directory) {
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        printAllFiles(file);
                    } else {
                    }
                }
            }
        } else {
        }
    }

    private void printAllAssets(String path) {
        try {
            AssetManager assetManager = context.getAssets();
            String[] files = assetManager.list(path);
            if (files != null) {
                for (String file : files) {
                    String filePath = path.isEmpty() ? file : path + "/" + file;
                    if (assetManager.list(filePath).length > 0) {
                        printAllAssets(filePath);
                    } else {
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to list assets in path: " + path, e);
        }
    }

    // *** TEST CODE ***
    private short[] readPcm16LeRawAsset(String assetName) throws IOException {
        AssetManager am = context.getAssets();
        try (InputStream in = am.open(assetName);
             BufferedInputStream bis = new BufferedInputStream(in);
             ByteArrayOutputStream baos = new ByteArrayOutputStream(32 * 1024)) {

            byte[] buf = new byte[8192];
            int n;
            while ((n = bis.read(buf)) != -1) {
                baos.write(buf, 0, n);
            }
            byte[] all = baos.toByteArray();
            if ((all.length & 1) != 0) {
                throw new IOException("Asset RAW has odd byte length: " + assetName + " (" + all.length + " bytes)");
            }
            int samples = all.length / 2;
            short[] pcm = new short[samples];

            for (int i = 0, s = 0; i < all.length; i += 2, s++) {
                int lo = (all[i] & 0xFF);
                int hi = (all[i + 1] << 8);
                pcm[s] = (short)(hi | lo);
            }
            return pcm;
        }
    }

    private int getAssetSampleCount(String assetName) throws IOException {
        AssetManager am = context.getAssets();
        try (InputStream in = am.open(assetName)) {
            int bytes = in.available();
            if ((bytes & 1) != 0) throw new IOException("Odd byte length for RAW asset: " + assetName);
            return bytes / 2;
        }
    }

    private double getAssetDurationSeconds(String assetName) throws IOException {
        int samples = getAssetSampleCount(assetName);
        return samples / 16000.0;
    }

    public void runExternalOnRawAsset_AllAtOnce(String assetName, boolean skipFirst2s) {
        try {
            if (!isListening || !isExternalMode.get()) {
                startListeningExternalAudio(/*threshold*/ 0.0f);
            }
            short[] pcm = readPcm16LeRawAsset(assetName);

            int offsetSamples = skipFirst2s ? (2 * 16000) : 0;
            if (offsetSamples < 0) offsetSamples = 0;
            if (offsetSamples > pcm.length) offsetSamples = pcm.length;

            if (offsetSamples == 0) {
                pushNextFrame(pcm, pcm.length);
            } else {
                int remain = pcm.length - offsetSamples;
                short[] tail = new short[remain];
                System.arraycopy(pcm, offsetSamples, tail, 0, remain);
                pushNextFrame(tail, tail.length);
            }
            Log.d(TAG, "runExternalOnRawAsset_AllAtOnce(): pushed '" + assetName + "', samples=" + pcm.length);
        } catch (Throwable e) {
            Log.e(TAG, "runExternalOnRawAsset_AllAtOnce failed for " + assetName, e);
            stopListening();
        }
    }
    
    public void runExternalOnRawAsset_Stream(String assetName, boolean skipFirst2s) {
        final int F = Constants.FRAME_LENGTH * 8;
        AssetManager am = context.getAssets();

        try (InputStream in = am.open(assetName);
             BufferedInputStream bis = new BufferedInputStream(in)) {

            if (!isListening || !isExternalMode.get()) {
                startListeningExternalAudio(/*threshold*/ 0.0f);
            }

            int toSkipBytes = skipFirst2s ? (2 * 16000 * 2) : 0;
            while (toSkipBytes > 0) {
                long skipped = bis.skip(toSkipBytes);
                if (skipped <= 0) break;
                toSkipBytes -= skipped;
            }

            byte[] b = new byte[F * 2];
            for (;;) {
                int off = 0, n;
                while (off < b.length && (n = bis.read(b, off, b.length - off)) > 0) off += n;
                if (off <= 0) break;

                int samples = off / 2;
                short[] chunk = new short[samples];
                for (int i = 0, s = 0; i + 1 < off; i += 2, s++) {
                    int lo = (b[i] & 0xFF);
                    int hi = (b[i + 1] << 8);
                    chunk[s] = (short)(hi | lo);
                }
                pushNextFrame(chunk, chunk.length);

                if (off < b.length) break;
            }
            Log.d(TAG, "runExternalOnRawAsset_Stream(): streamed '" + assetName + "'");
        } catch (Throwable e) {
            Log.e(TAG, "runExternalOnRawAsset_Stream failed for " + assetName, e);
            stopListening();
        }
    }

    public void testTwoRawAssets() {
        startListeningExternalAudio(/*threshold*/ 0.0f);

        runExternalOnRawAsset_AllAtOnce("audio_1758288269102.raw", /*skipFirst2s*/ false);
        //runExternalOnRawAsset_AllAtOnce("blabla.raw", false);

        // runExternalOnRawAsset_Stream("audio_1758288269102.raw", true);
        // runExternalOnRawAsset_Stream("blabla.raw", true);
    }
    // END OF TEST CODE

}
