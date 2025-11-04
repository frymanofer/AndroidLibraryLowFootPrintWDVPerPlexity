
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

    private String vadPath = null;

    private OrtEnvironment env = null;
    private OrtSession[] sessions;
    private OrtSession melspecSession;
    private final String TAG = "KeyWordsDetection";
    private Context context;
    private Map<String, OnnxTensor> inputs;
    private List<String>[] inputNames;
    private int[] nFeatureFrames;
    private volatile boolean isListening;
    private int bufferSize = 0;
    private Set<String> melspecInputNames;
    private static final int MEL_SPECTROGRAM_MAX_LEN = 10 * 97;
    private static final int SAMPLE_RATE = 16000; // currently
    private static final int RAW_BUFFER_MAX_LEN = (1280 * 2);
    // tail padding (ms) used by both v1 and v2 external full-buffer APIs
    private static final int HEAD_PAD_MS = 1000;
    private static final int TAIL_PAD_MS = 1000;
    // 2-second keyword clip length used during training (from Python kwd_clip_length)
    private static final int V2_KWD_CLIP_SAMPLES = 31712;

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
    private int[] perModellPredictions;
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
    private short[] zeroArr = new short[Constants.FRAME_LENGTH * 2];

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

    private final ArrayList<float[][]> bulkWindows = new ArrayList<>(512);
    private boolean bulkCollecting = false; // true while ingesting the "head" of a bulk push

    private int melspecInputRank = 1; // 1D or 2D (batch, time)
    private static final float INV_SHORT_MAX = 1.0f / 32768.0f;

    // Reusable tiny objects to reduce churn
    // === Timing helpers (added) ===
    private static long tNow() { return System.nanoTime(); }
    private static String tMs(long t0) {
        return String.format(Locale.US, "%.2f ms", (System.nanoTime() - t0) / 1_000_000.0);
    }
    private static final boolean DEBUG = false;
    
    static {
        try {
            System.loadLibrary("onnxruntime");
            //Log.d("KeyWordsDetection", "onnxruntime loaded successfully.");
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
            //Log.d("KeyWordsDetection", "onnxruntime4j_jni loaded successfully.");
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

        Log.d(TAG, "KeyWordsDetection constructor: ");
        this.keyThreasholds = thresholds;
        this.keyBufferCnts = bufferCnts;
        this.msBetweenCallbacks = msBetweenCallback;

        int numModels = modelPaths.length;
        fakeThresholds = new float[numModels];
        lastCallbackInMS = new long[numModels];
        concurrentPredictions = new int[numModels];
        perModellPredictions = new int[numModels];
        for (int i = 0; i < numModels; i++) {
            fakeThresholds[i] = thresholds[i] - 0.1f;
            lastCallbackInMS[i] = 0;
            concurrentPredictions[i] = 0;
            perModellPredictions[i] = 0;
        }

        Log.d(TAG, "KeyWordsDetection constructor: keyThreasholds: " + keyThreasholds);
        Log.d(TAG, "KeyWordsDetection constructor: fakeThresholds: " + fakeThresholds);
        audioBuffer = new short[LAST_SEC_BUFF_SIZE];
        Arrays.fill(audioBuffer, (short) 0);

        Arrays.fill(zeroArr, (short) 0);
        Log.d(TAG, "KeyWordsDetection constructor: keyBufferCnts: " + keyBufferCnts);
        inputs = new HashMap<>();

        this.context = context;
        strippedModelNames = new String[modelPaths.length];
        internalModelPaths = new String[modelPaths.length];

        Boolean isPlex = false;

        Log.d(TAG, "KeyWordsDetection constructor: modelPaths.length: " + modelPaths.length);

        for (int i = 0; i < modelPaths.length; i++) {
            String name = modelPaths[i];
            name = name.substring(name.lastIndexOf('/') + 1);
            name = name.replaceFirst("[.][^.]+$", "");
            strippedModelNames[i] = name;

            String local = assetExists(modelPaths[i])
                 ? copyAssetToInternalStorage(modelPaths[i])
                 : modelPaths[i];
            internalModelPaths[i] = local;
            logOnnxAndData(internalModelPaths[i]);
        }
        String melspecPath;
        melspecPath   = copyAssetIfExists("melspectrogram.onnx");
 
        Log.i(TAG, "internalModelPaths[0] == " + internalModelPaths[0]);
        if (internalModelPaths[0] == null || melspecPath == null) {
            Log.e(TAG, "KeyWordsDetection: Model path is null after copying asset.");
            throw new OrtException("Failed to copy model asset to internal storage.");
        }

        try {
            //Log.d(TAG, "KeyWordsDetection constructor: Models: " + Arrays.toString(modelPaths));
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();

            final int cores = Math.max(1, Runtime.getRuntime().availableProcessors());

            try {
                Map<String, String> xnnpackOpts = Collections.emptyMap();
                options.addXnnpack(xnnpackOpts);
                Log.i(TAG, "XNNPACK EP added successfully.");
            } catch (Exception e) {
                Log.w(TAG, "XNNPACK EP not available or failed to load.", e);
            }

            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
//            options.setIntraOpNumThreads(Math.min(2, cores)); Best????
            options.setIntraOpNumThreads(Math.min(2, cores)); 
            options.setInterOpNumThreads(1);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
            options.setCPUArenaAllocator(true);
            options.setMemoryPatternOptimization(true);
            options.setSessionLogVerbosityLevel(0);

            try {
                melspecSession = env.createSession(melspecPath, options);
            } catch (Exception e) {
                Log.e(TAG, "Failed to create melspecSession from " + melspecPath, e);
                throw e;
            }
// IMPORTANT: initialize melspecInputNames before using it
try {
    melspecInputNames = melspecSession.getInputNames();
    Map<String, NodeInfo> melInputInfoMap = melspecSession.getInputInfo();

    // pick the first input name
    String melInputName = melspecInputNames.iterator().next();
    TensorInfo melTensorInfo = (TensorInfo) melInputInfoMap.get(melInputName).getInfo();
    long[] melShape = melTensorInfo.getShape();

    melspecInputRank = (melShape == null) ? 1 : melShape.length;
    Log.i(TAG, "melspec input rank = " + melspecInputRank +
            " shape=" + Arrays.toString(melShape));
} catch (Exception e) {
    Log.w(TAG, "Could not read melspec input shape; defaulting to rank 2 [1, N]", e);
    // If you KNOW your exported model is [1, N], it's safer to default to 2:
    melspecInputRank = 2;
}

            melspecInputNames = melspecSession.getInputNames();

            rawDataBuffer = new ArrayDeque<>(RAW_BUFFER_MAX_LEN);
            melspectrogramBuffer = new ArrayList<>();

            sessions = new OrtSession[modelPaths.length];
            inputNames      = new ArrayList[modelPaths.length];
            nFeatureFrames  = new    int  [modelPaths.length];
            for (int i = 0; i < modelPaths.length; i++) {
                                try {
                Log.i(TAG, "Creating KWS session from: " + internalModelPaths[i]);
                    sessions[i] = env.createSession(internalModelPaths[i], options);
                } catch (Exception e) {
                    Log.e(TAG, "Failed to create KWS session[" + i + "] from "
                            + internalModelPaths[i], e);
                    throw e;
                }

                // sessions[i] = env.createSession(internalModelPaths[i], options);
                inputNames[i] = new ArrayList<>(sessions[i].getInputNames());
                Map<String, NodeInfo> inputInfoMap = sessions[i].getInputInfo();
                NodeInfo inputInfo = inputInfoMap.get(inputNames[i].get(0));
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();
                nFeatureFrames[i] = (int) shape[1];
                //Log.d(TAG, "Model [" + strippedModelNames[i] + "] nFeatureFrames: " + nFeatureFrames[i]);
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

            randomData = new short[randomDataSize]; // zeroes!!!
            featureDeque.clear();
            int initLen = featureBufferMaxLen; // typically 120
            for (int i = 0; i < initLen; i++) {
                float[] row = new float[96];                  // Java initializes to 0.0f by default
                updateFeatureQueue(row);
            }
            warmupModels();

        } catch (OrtException e) {
            e.printStackTrace();
            throw e;
        }
    }

        private void logOnnxAndData(String onnxPath) {
        try {
            File onnx = new File(onnxPath);
            File data = new File(onnxPath + ".data");

            Log.i(TAG, "ONNX model path: " + onnx.getAbsolutePath()
                    + " exists=" + onnx.exists() + " size=" + (onnx.exists() ? onnx.length() : -1));

            Log.i(TAG, "ONNX external data path: " + data.getAbsolutePath()
                    + " exists=" + data.exists() + " size=" + (data.exists() ? data.length() : -1));
        } catch (Throwable t) {
            Log.e(TAG, "logOnnxAndData failed", t);
        }
    }


    public void close() {
        //Log.d(TAG, "KeyWordsDetection close() called, cleaning up all native resources.");

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
        //Log.d(TAG, "KeyWordsDetection close() completed.");
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
                //Log.d("File Search", "File found: " + file.getAbsolutePath());
                return copyFileToExternalStorage(file.getAbsolutePath());
            } else {
                //Log.d("File Search", "File not found in: " + dir);
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
            //Log.d("copyFileToExternal", "File copied to: " + destFile.getAbsolutePath());
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
        //Log.d(TAG, "replaceKeywordDetectionModel()");
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
            //Log.d(TAG, "KeyWordsDetection constructor: Models: " + Arrays.toString(modelPaths));
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
                //Log.d(TAG, "Model [" + strippedModelNames[i] + "] nFeatureFrames: " + nFeatureFrames[i]);
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

    private String copyAssetToInternalStorage_v1(String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        File parentFile = file.getParentFile();
        if (!parentFile.exists()) {
            boolean dirsCreated = parentFile.mkdirs();
        }

        if (file.exists()) {
            boolean deleted = file.delete();
            //Log.d(TAG, "Existing asset deleted: " + deleted);
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

    private String copyAssetToInternalStorage(String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        File parentFile = file.getParentFile();
        if (!parentFile.exists()) {
            boolean dirsCreated = parentFile.mkdirs();
        }

        if (file.exists()) {
            boolean deleted = file.delete();
            //Log.d(TAG, "Existing asset deleted: " + deleted);
        }

        AssetManager assetManager = context.getAssets();

        // 1) Copy the main asset (e.g. hey_plex.onnx)
        if (!file.exists()) {
            try {
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

        // 2) If this is an ONNX file, also copy the external data file "<name>.onnx.data" if it exists.
        //    ONNX Runtime will automatically pick it up as long as it sits next to the .onnx.
        if (assetName.endsWith(".onnx")) {
            String dataAssetName = assetName + ".data";  // e.g. "hey_plex.onnx.data" or "models/hey_plex.onnx.data"
            if (assetExists(dataAssetName)) {
                File dataFile = new File(context.getFilesDir(), dataAssetName);
                File dataParent = dataFile.getParentFile();
                if (!dataParent.exists()) {
                    boolean created = dataParent.mkdirs();
                }

                if (dataFile.exists()) {
                    boolean deletedData = dataFile.delete();
                    //Log.d(TAG, "Existing .onnx.data deleted: " + deletedData);
                }

                try {
                    InputStream din = assetManager.open(dataAssetName);
                    FileOutputStream dout = new FileOutputStream(dataFile);
                    byte[] dbuf = new byte[1024];
                    int dread;
                    while ((dread = din.read(dbuf)) != -1) {
                        dout.write(dbuf, 0, dread);
                    }
                    din.close();
                    dout.close();
                    //Log.d(TAG, "Copied ONNX external data asset: " + dataAssetName);
                } catch (IOException e) {
                    Log.e(TAG, "Failed to copy ONNX .data asset file: " + dataAssetName, e);
                    // We don't return null here because the main .onnx is present; you can decide how strict you want to be.
                }
            } else {
                // Optional: log if .data is missing; ORT will still work if the model is not using external data.
                //Log.d(TAG, "No ONNX .data asset found for: " + assetName);
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
        //Log.d(TAG, "Saving WAV file: " + fileName);

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
            //Log.d(TAG, "lastFile: " + lastFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void ignoreBatteryOptimization() {
        Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:" + context.getPackageName()));
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(intent);
    }

// One-time warmup for melspec + KWS models
private void warmupModels() {
    try {
        // silent 2-second clip, same length as training
        short[] silent = new short[V2_KWD_CLIP_SAMPLES];

        // Warm up melspectrogram ONNX
        float[][] mel = getMelspectrogramForV2(silent); // [T, F]
        if (mel == null || mel.length == 0 || mel[0] == null) {
            Log.w(TAG, "warmupModels: melspec returned empty output");
            return;
        }

        // Warm up each KWS ONNX session once
        if (sessions != null) {
            for (int i = 0; i < sessions.length; i++) {
                try {
                    runKwsModelOnMel(i, mel);
                } catch (Exception e) {
                    Log.w(TAG, "warmupModels: KWS session[" + i + "] warmup failed", e);
                }
            }
        }
        Log.i(TAG, "warmupModels: warmup completed");
    } catch (Throwable t) {
        Log.w(TAG, "warmupModels: overall warmup failed", t);
    }
}

    public void startListeningExternalAudio(float threshold) {
        if (isListening) {
            //Log.d(TAG, "Already listening");
            return;
        }
        Arrays.fill(zeroArr, (short) 0);

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

            featureDeque.clear();
            int initLen = featureBufferMaxLen; // typically 120
            for (int i = 0; i < initLen; i++) {
                float[] row = new float[96];  // 96 = embedding/feature dim
                // Java initializes to 0.0f by default
                updateFeatureQueue(row);
            }
            warmupModels();

            extCarryLen = 0;

            //Log.d(TAG, "startListeningExternalAudio(): init complete");
            //Log.d(TAG, "keyThreasholds: " + Arrays.toString(keyThreasholds));
            //Log.d(TAG, "fakeThresholds: " + Arrays.toString(fakeThresholds));
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
        bulkMinSamples = Math.max(1280 * 2, samples);
    }

    // Wrapper: by default we use v2 (direct mel->ONNX KWS). You can flip to _v1 if needed.
    public boolean predictFromExternalFullBuffer(short[] pcm, int length) {
        return predictFromExternalFullBuffer_v2(pcm, length);
        // If you want the old behaviour:
        // return predictFromExternalFullBuffer_v1(pcm, length);
    }
    
// v2 helper: run ONNX KWS model directly on mel [T,F] -> probability (positive class)
private float runKwsModelOnMel(int modelIndex, float[][] mel) {
    if (sessions == null || modelIndex < 0 || modelIndex >= sessions.length) {
        return 0.0f;
    }
    if (mel == null || mel.length == 0 || mel[0] == null) {
        return 0.0f;
    }

    int T = mel.length;
    int F = mel[0].length;

    OnnxTensor inputTensor = null;
    OrtSession.Result result = null;
    Map<String, OnnxTensor> localInputs = null;

    try {
        String inputName = inputNames[modelIndex].get(0);
        Map<String, NodeInfo> inputInfoMap = sessions[modelIndex].getInputInfo();
        TensorInfo tinfo = (TensorInfo) inputInfoMap.get(inputName).getInfo();
        long[] modelShape = tinfo.getShape();
        int rank = (modelShape != null ? modelShape.length : 0);

        float[][] melAdj = mel;  // may adjust T to match model
        long[] shape;

        if (rank == 3) {
            // (B, T_model, F_model)
            long tModel = modelShape[1];
            long fModel = modelShape[2];

            if (fModel > 0 && fModel != F) {
                Log.e(TAG, "runKwsModelOnMel: mel F mismatch (model F=" + fModel + ", got " + F + ")");
                return 0.0f;
            }

            if (tModel > 0 && tModel != T) {
                int targetT = (int) tModel;
                float[][] newMel = new float[targetT][F];

                if (T >= targetT) {
                    // keep last targetT frames
                    int start = T - targetT;
                    for (int i = 0; i < targetT; i++) {
                        newMel[i] = mel[start + i];
                    }
                } else {
                    // pad at front with zeros, place mel at the end
                    int pad = targetT - T;
                    for (int i = 0; i < pad; i++) {
                        newMel[i] = new float[F]; // zeros
                    }
                    for (int i = 0; i < T; i++) {
                        newMel[pad + i] = mel[i];
                    }
                }
                melAdj = newMel;
                T = targetT;
            }

            float[] flattened = flatten(melAdj);
            shape = new long[]{1, T, F};
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flattened), shape);
        } else if (rank == 4) {
            // (B, C, T_model, F_model) with C=1
            long cModel = modelShape[1];
            long tModel = modelShape[2];
            long fModel = modelShape[3];

            if (cModel != 1 && cModel > 0) {
                Log.e(TAG, "runKwsModelOnMel: unexpected channel dim (shape[1]=" + cModel + ")");
                return 0.0f;
            }
            if (fModel > 0 && fModel != F) {
                Log.e(TAG, "runKwsModelOnMel: mel F mismatch (model F=" + fModel + ", got " + F + ")");
                return 0.0f;
            }

            if (tModel > 0 && tModel != T) {
                int targetT = (int) tModel;
                float[][] newMel = new float[targetT][F];

                if (T >= targetT) {
                    // keep last targetT frames
                    int start = T - targetT;
                    for (int i = 0; i < targetT; i++) {
                        newMel[i] = mel[start + i];
                    }
                } else {
                    // pad at front with zeros, place mel at the end
                    int pad = targetT - T;
                    for (int i = 0; i < pad; i++) {
                        newMel[i] = new float[F];
                    }
                    for (int i = 0; i < T; i++) {
                        newMel[pad + i] = mel[i];
                    }
                }
                melAdj = newMel;
                T = targetT;
            }

            float[] flattened = flatten(melAdj);
            shape = new long[]{1, 1, T, F}; // (B, C=1, T, F)
            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flattened), shape);
        } else {
            Log.e(TAG, "runKwsModelOnMel: unsupported input rank " + rank);
            return 0.0f;
        }

        localInputs = new HashMap<>();
        localInputs.put(inputName, inputTensor);

        result = sessions[modelIndex].run(localInputs);
        Object out = result.get(0).getValue();

        // logits -> probability of positive class
        if (out instanceof float[][]) {
            float[][] logits2D = (float[][]) out;
            if (logits2D.length == 0 || logits2D[0] == null || logits2D[0].length == 0) {
                return 0.0f;
            }
            float[] logits = logits2D[0];
            if (logits.length == 1) return logits[0];

            float l0 = logits[0];
            float l1 = logits[1];
            float m = Math.max(l0, l1);
            float e0 = (float) Math.exp(l0 - m);
            float e1 = (float) Math.exp(l1 - m);
            return e1 / (e0 + e1);
        } else if (out instanceof float[]) {
            float[] logits = (float[]) out;
            if (logits.length == 0) return 0.0f;
            if (logits.length == 1) return logits[0];

            float l0 = logits[0];
            float l1 = logits[1];
            float m = Math.max(l0, l1);
            float e0 = (float) Math.exp(l0 - m);
            float e1 = (float) Math.exp(l1 - m);
            return e1 / (e0 + e1);
        } else {
            Log.w(TAG, "runKwsModelOnMel: unexpected output type " + out.getClass());
            return 0.0f;
        }
    } catch (Exception e) {
        Log.e(TAG, "runKwsModelOnMel failed: " + e.getMessage());
        return 0.0f;
    } finally {
        if (localInputs != null) {
            localInputs.clear();
        }
        if (result != null) result.close();
        if (inputTensor != null) inputTensor.close();
    }
}

    // small helper: logits -> prob(class 1)
    private float softmaxPositive(float[] logits) {
        if (logits == null || logits.length == 0) return 0.0f;
        if (logits.length == 1) return logits[0]; // model already outputs prob/logit for positive class

        float l0 = logits[0];
        float l1 = logits[1];
        float m = Math.max(l0, l1);
        float e0 = (float) Math.exp(l0 - m);
        float e1 = (float) Math.exp(l1 - m);
        return e1 / (e0 + e1);
    }

// v2-specific: melspec from ONNX WITHOUT the /10 + 2 transform
private float[][] getMelspectrogramForV2(short[] audioData) {
    float[] audioDataFloat = new float[audioData.length];
    for (int i = 0; i < audioData.length; i++) {
        // same normalization as Python waveform
        audioDataFloat[i] = audioData[i] * INV_SHORT_MAX;
    }

    long[] inputShape;
    if (melspecInputRank == 1) {
        // Python: waveform.numpy() 1D
        inputShape = new long[]{audioDataFloat.length};
    } else {
        // Python: waveform.numpy() 2D [1, N]
        inputShape = new long[]{1, audioDataFloat.length};
    }
    OnnxTensor inputTensor = null;
    OrtSession.Result result = null;
    String melSpecInputName = null;
    Map<String, OnnxTensor> localInputs = null;

    try {
        inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audioDataFloat), inputShape);
        localInputs = new HashMap<>();
        melSpecInputName = melspecInputNames.iterator().next();
        localInputs.put(melSpecInputName, inputTensor);

        result = melspecSession.run(localInputs);
        Object output = result.get(0).getValue();

        // Same helper you already have
        return coerceMelTo2D(output);  // [T,F]
    } catch (Exception e) {
        Log.e(TAG, "getMelspectrogramForV2: failed to run melspec session: " + e.getMessage());
        return new float[0][0];
    } finally {
        if (localInputs != null && melSpecInputName != null) {
            localInputs.remove(melSpecInputName);
        }
        if (result != null) result.close();
        if (inputTensor != null) inputTensor.close();
    }
}

    private boolean processOneKwFrameV2(short[] frame, int frameLength) {
        try {
            // 1) update raw buffer (same as other paths)
            bufferRawData(frame);
            if (rawDataBuffer.size() < MIN_COLLECT_SAMPLES) {
                return false; // not enough audio yet
            }

            // 2) streaming mel update using existing logic
            streamingMelspectrogram(frameLength);

            int ndx = getMelNumberOfRows();
            if (ndx < 76) {
                return false; // need at least 76 mel frames
            }

            // 3) take last 76×32 window
            float[][][][] melspectrogramSlice = getMelspecSubArray(ndx - 76, ndx);
            float[][] melWindow = flattenSubArray(melspectrogramSlice); // [76, 32]

            if (sessions == null || sessions.length == 0) {
                return false;
            }

            boolean detected = false;

            // 4) run each ONNX KWS model directly on this mel window
            for (int i = 0; i < sessions.length; i++) {
                float meanPrediction = runKwsModelOnMel(i, melWindow);

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
                                detected = true;
                            }
                            concurrentPredictions[i] = 0;
                        }
                    }
                } else {
                    concurrentPredictions[i] = 0;
                }
            }
            return detected;
        } catch (Throwable fatal) {
            Log.e(TAG, "processOneKwFrameV2 failed – stopping detector", fatal);
            stopListening();
            return false;
        }
    }

// === v2: streaming over full buffer + head/tail pad -> melspec ONNX -> KWS ONNX ===
// === v2: match Python eval_wav_fixed_windows (PCM sliding) ===
public boolean predictFromExternalFullBuffer_v2(short[] pcm, int length) {
    if (!isListening || !isExternalMode.get() || pcm == null || length <= 0) return false;

    final int SR          = SAMPLE_RATE;             // 16000
    final int STRIDE_SAMP = Constants.FRAME_LENGTH;  // 1280
    final int KWD_CLIP    = V2_KWD_CLIP_SAMPLES;     // 31712

    // Keep WAV debug behaviour (3s ring buffer)
    int storeOffset = 0;
    while (storeOffset < length) {
        int frameLen = Math.min(STRIDE_SAMP, length - storeOffset);
        short[] frame = new short[frameLen];
        System.arraycopy(pcm, storeOffset, frame, 0, frameLen);
        storeFrame(frame, frameLen);
        storeOffset += frameLen;
    }

    long t0 = tNow();

    try {
        // ---- 1) Build wav buffer like Python eval_wav_fixed_windows ----
        int total = length;

        // If shorter than KWD_CLIP: pad at FRONT
        int frontPad = 0;
        if (total < KWD_CLIP) {
            frontPad = KWD_CLIP - total;
            total = KWD_CLIP;
        }

        // Tail pad = stride * 30 (same as Python: pad_amount = stride * 30)
        int tailPad = STRIDE_SAMP * 30;
        total += tailPad;

        short[] wav = new short[total];
        // frontPad samples stay zero
        System.arraycopy(pcm, 0, wav, frontPad, length);
        // tailPad zeros at the end

        int T_total = total;
        int nSteps  = 1 + (int) Math.floor((T_total - KWD_CLIP) / (double) STRIDE_SAMP);
        if (nSteps <= 0) {
            Log.w(TAG, "predictFromExternalFullBuffer_v2: nSteps <= 0");
            return false;
        }

        if (sessions == null || sessions.length == 0) return false;

        boolean detected = false;

        // ---- 2) Slide PCM windows exactly like Python ----
        for (int step = 0;
             step < nSteps && isListening && isExternalMode.get() && !detected;
             step++) {

            int windowStart = step * STRIDE_SAMP;
            int windowEnd   = windowStart + KWD_CLIP;

            if (windowEnd > T_total) break;

            // Extract PCM chunk [KWD_CLIP] samples
            short[] chunk = new short[KWD_CLIP];
            System.arraycopy(wav, windowStart, chunk, 0, KWD_CLIP);

            // ---- 3) ONNX melspec for this chunk (v2: NO /10+2 here) ----
            float[][] mel = getMelspectrogramForV2(chunk); // [T,F]
            if (mel == null || mel.length == 0 || mel[0] == null) {
                continue;
            }

            // ---- 4) Run each KWS ONNX model on this mel window ----
            for (int i = 0; i < sessions.length; i++) {
                float meanPrediction = runKwsModelOnMel(i, mel);
                // Log.d(TAG, "v2 step=" + step + " t=" +
                //         (windowStart / (float) SR) + "s model " + i +
                //         " meanPrediction = " + meanPrediction);

                if (meanPrediction > fakeThresholds[i]) {
                    if (meanPrediction < keyThreasholds[i]) {
                        concurrentPredictions[i] = 0;
                    } else {
                        concurrentPredictions[i]++;
                        if (concurrentPredictions[i] >= keyBufferCnts[i]) {
                            long now = System.currentTimeMillis();
                            if (lastCallbackInMS[i] + msBetweenCallbacks[i] <= now) {
                                lastCallbackInMS[i] = now;
                                String fileName = strippedModelNames[i] + "_prediction_v2.wav";
                                flushBufferToWav(fileName);
                                if (keywordDetectedCallback != null) {
                                    keywordDetectedCallback.accept(true, strippedModelNames[i]);
                                }
                                detected = true;
                            }
                            concurrentPredictions[i] = 0;
                        }
                    }
                } else {
                    concurrentPredictions[i] = 0;
                }

                if (detected) break;
            }
        }

        Log.d(TAG, "predictFromExternalFullBuffer_v2: " +
                (detected ? "Predicted after time: " : "False reported after time: ") + tMs(t0));
        return detected;
    } catch (Throwable fatal) {
        Log.e(TAG, "predictFromExternalFullBuffer_v2 failed – stopping detector", fatal);
        stopListening();
        return false;
    }
}

    public void startListening(float threshold) throws OrtException {

        if (isListening) {
            //Log.d(TAG, "Already listening");
            return;
        }
        Log.e(TAG, "Start Listening with Mic not implemented!!!");
    }

    public void stopListening() {
        if (!isListening) {
            //Log.d(TAG, "Stop listening is called while not listening.");
            return;
        }
        isListening = false;

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
    private static final int MIN_COLLECT_SAMPLES = 1280;

    private void detectHeadCollect(short[] audioData) throws OrtException {
        //Log.d(TAG,"detectHeadCollect()");
        bufferRawData(audioData);
        if (rawDataBuffer.size() < MIN_COLLECT_SAMPLES) {
            //Log.d(TAG,"detectHeadCollect() rawDataBuffer.size() < MIN_COLLECT_SAMPLES");
            // Not enough audio yet; keep accumulating
            return;
        }

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
        // //Log.d(TAG, "streamingMelspectrogram: " + tMs(t0) + " (nSamples=" + nSamples + ", raw=" + rawData.length + ")");
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
            // match Python torchaudio waveform ~[-1, 1]
            audioDataFloat[i] = audioData[i] * INV_SHORT_MAX;
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

// Normalize ONNX mel output to 2D [T, F] without OnnxTensor casts (Java 8)
private static float[][] coerceMelTo2D(Object v) {
    if (v instanceof float[][]) {
        return (float[][]) v;
    } else if (v instanceof float[][][]) {
        float[][][] m3 = (float[][][]) v;
        int a = m3.length, b = (a>0? m3[0].length:0), c = (b>0? m3[0][0].length:0);
        // Prefer [1,T,F] -> T=b, F=c
        if (a == 1 && b > 1 && c > 1) return m3[0];
        // Handle [T,F,1] -> drop last singleton
        if (c == 1 && a > 0 && b > 0) {
            float[][] out = new float[a][b];
            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                    out[i][j] = m3[i][j][0];
            return out;
        }
        // Fallback: choose F from common set
        return coerceByGuessingTF3D(m3);
} else if (v instanceof float[][][][]) {
    float[][][][] m4 = (float[][][][]) v;
    int a = m4.length;
    int b = (a>0? m4[0].length:0);
    int c = (b>0? m4[0][0].length:0);
    int d = (c>0? m4[0][0][0].length:0);

    // [1,1,T,F]  -> ok even if T==1
    if (a == 1 && b == 1 && c >= 1 && d >= 1) {
        float[][] out = new float[c][d];
        for (int i = 0; i < c; i++)
            System.arraycopy(m4[0][0][i], 0, out[i], 0, d);
        return out;
    }
    // [1,T,F,1]  -> ok even if T==1
    if (a == 1 && b >= 1 && c >= 1 && d == 1) {
        float[][] out = new float[b][c];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < c; j++)
                out[i][j] = m4[0][i][j][0];
        return out;
    }

    // Generic singleton permutations: pick F from {32,40,64,80,96}, T = any other dim >=1
    int[] dims = new int[]{a,b,c,d};
    int[] commonF = new int[]{32,40,64,80,96};
    int fIdx = -1, F = -1;
    for (int idx = 0; idx < 4 && fIdx==-1; idx++) {
        for (int cf : commonF) {
            if (dims[idx] == cf) { fIdx = idx; F = cf; break; }
        }
    }
    if (fIdx != -1) {
        int tIdx = -1, T = -1;
        for (int idx = 0; idx < 4; idx++) {
            if (idx == fIdx) continue;
            int val = dims[idx];
            if (val >= 1 && val > T) { T = val; tIdx = idx; }
        }
        if (T >= 1) {
            float[][] out = new float[T][F];
            for (int ti = 0; ti < T; ti++) {
                for (int fj = 0; fj < F; fj++) {
                    out[ti][fj] = get4(m4, tIdx, ti, fIdx, fj);
                }
            }
            return out;
        }
    }

    // Last resort: flatten to 1D then reshape if possible (handles very odd exports)
    int N = a*b*c*d;
    float[] flat = new float[N];
    int idx = 0;
    for (int i=0;i<a;i++)
        for (int j=0;j<b;j++)
            for (int k=0;k<c;k++)
                for (int l=0;l<d;l++)
                    flat[idx++] = m4[i][j][k][l];
    int[] common = new int[]{32,40,64,80,96};
    for (int CF: common) {
        if (N % CF == 0) {
            int T = N / CF;
            float[][] out = new float[T][CF];
            int p = 0;
            for (int i=0;i<T;i++) {
                System.arraycopy(flat, p, out[i], 0, CF);
                p += CF;
            }
            return out;
        }
    }
    throw new IllegalArgumentException("Unsupported 4D mel layout: ["+a+","+b+","+c+","+d+"]");
} else if (v instanceof float[]) {
        float[] m1 = (float[]) v; // flat vector; try to infer F and T
        int N = m1.length;
        int[] commonF = new int[]{32,40,64,80,96};
        for (int F : commonF) {
            if (N % F == 0) {
                int T = N / F;
                float[][] out = new float[T][F];
                int idx = 0;
                for (int i = 0; i < T; i++) {
                    System.arraycopy(m1, idx, out[i], 0, F);
                    idx += F;
                }
                return out;
            }
        }
        throw new IllegalArgumentException("MelSpec 1D cannot be reshaped; len="+N);
    } else {
        throw new IllegalArgumentException("Unsupported mel type: " + v.getClass());
    }
}

// Helper to read value from arbitrary 4D layout by logical T and F indices.
private static float get4(float[][][][] m4, int tIdx, int t, int fIdx, int f) {
    // map (tIdx, fIdx) -> indices (a,b,c,d). Any remaining singleton dims -> 0
    int a = 0, b = 0, c = 0, d = 0;
    if (tIdx == 0) a = t; else if (fIdx == 0) a = f;
    if (tIdx == 1) b = t; else if (fIdx == 1) b = f;
    if (tIdx == 2) c = t; else if (fIdx == 2) c = f;
    if (tIdx == 3) d = t; else if (fIdx == 3) d = f;
    return m4[a][b][c][d];
}

// Fallback for odd 3D layouts
private static float[][] coerceByGuessingTF3D(float[][][] m3) {
    int a = m3.length, b = (a>0? m3[0].length:0), c = (b>0? m3[0][0].length:0);
    int[] commonF = new int[]{32,40,64,80,96};
    // try (T=a,F=b)
    for (int cf: commonF) if (b==cf && a>1 && c==1) {
        float[][] out = new float[a][b];
        for (int i=0;i<a;i++) System.arraycopy(m3[i][0], 0, out[i], 0, b);
        return out;
    }
    // try (T=b,F=c) when a==1 handled earlier
    // try (T=a,F=c) with b==1
    if (b==1 && c>1 && a>1) {
        float[][] out = new float[a][c];
        for (int i=0;i<a;i++)
            System.arraycopy(m3[i][0], 0, out[i], 0, c);
        return out;
    }
    throw new IllegalArgumentException("Unsupported 3D mel layout: ["+a+","+b+","+c+"]");
}

    private float[][] getMelspectrogram(float[] audioDataFloat) {
        //long t0 = tNow();
        //Log.d(TAG, "getMelspectrogram()");

        long[] inputShape;
        if (melspecInputRank == 1) {
            // Python: waveform.numpy() 1D
            inputShape = new long[]{audioDataFloat.length};
        } else {
            // Python: waveform.numpy() 2D [1, N]
            inputShape = new long[]{1, audioDataFloat.length};
        }
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
            ////Log.d(TAG, "melspecSession.run: " + tMs(tRun0));

            Object output = result.get(0).getValue();
            // float[][][][] melspectrogram4D = (float[][][][]) output;
            // float[][] melspectrogram = (float [][]) squeeze4D(melspectrogram4D);
            float[][] melspectrogram = coerceMelTo2D(output);
            
            //long tTf0 = tNow();
            Function<Float, Float> melspecTransform = x -> x / 10 + 2;
            float [][] spec = transform(melspectrogram, melspecTransform);
            ////Log.d(TAG, "melspec transform: " + tMs(tTf0));
            ////Log.d(TAG, "getMelspectrogram total: " + tMs(t0) + " (len=" + audioDataFloat.length + ")");
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

        for (int i = 0; i <= spec.length - windowSize; i += Constants.STEP_SIZE) {
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

    private void updateFeatureQueue(float[] newRow) {
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

}
