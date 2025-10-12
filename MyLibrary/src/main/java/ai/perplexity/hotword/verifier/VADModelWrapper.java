package ai.perplexity.hotword.verifier;

import ai.onnxruntime.*;
import android.content.Context;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.*;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import android.content.res.AssetManager;
import java.io.IOException;
import android.os.Process;
import java.util.Deque;

public class VADModelWrapper {
    private OrtEnvironment env;
    private OrtSession session;
    private final String TAG = "VADModelWrapper";
    private Context context;
    private int predict_call_times = 0;
   // private static final int BUFFER_SIZE = 40; // or any desired size
    private float[] h;
    private float[] c;
    long[] c_tensorShape;
    long[] h_tensorShape;
    Deque<Float> framePredictions;
    List<float[]> chunks;
    Map<String, OnnxTensor> inputs;
    List<String> inputNames;
    long[] srArray;
    long[] srShape;
    float[] floatInput;
    long[] chunkShape;

    private FloatBuffer inputBuffer;
    private FloatBuffer hBuffer;
    private FloatBuffer cBuffer;
    private LongBuffer srBuffer;
    OnnxTensor srTensor;
    OnnxTensor hTensor;
    OnnxTensor cTensor;

    public VADModelWrapper(Context context, String modelPath) throws OrtException {
        Log.d(TAG, "VADModelWrapper constructor: ");
        h = new float[128];
        c = new float[128];
        Arrays.fill(h, 0.0f); // Fill h with zeros
        Arrays.fill(c, 0.0f); // Fill c with zeros
        c_tensorShape = new long[]{2, 1, 64};
        h_tensorShape = new long[]{2, 1, 64};
        framePredictions = new ArrayDeque<>();
        // Initialize an empty list to hold the chunks
        inputs = new HashMap<>();
        srArray = new long[]{16000};
        srShape = new long[]{1};
        chunkShape = new long[]{1, Constants.FRAME_LENGTH};
        srBuffer = LongBuffer.wrap(new long[]{16000});
        floatInput = new float[Constants.FRAME_LENGTH];

        this.context = context;
        // Print debug to print all files in the bundle and all assets
        // printAllFilesInInternalStorage();


        // THIS IS WHERE THE MEMORY OVERRUN !!!!!!!!!!!!!!!!!!
        // If we return here all works well if not Mem overrun
        // RETURN ************
        // CHECK WITH LARGET STACK SIZE!!!
        modelPath = copyAssetToInternalStorage(modelPath);

        if (modelPath == null) {
            Log.e(TAG, "VADModelWrapper: Model path is null after copying asset.");
            throw new OrtException("Failed to copy model asset to internal storage.");
        }

        
        try {
            Log.d(TAG, "VADModelWrapper constructor: 1: " + modelPath);
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            // Nnapi may improve performance. Without it still 90+ % detections sometimes. 
            
            boolean nnapiAdded = true;
            
            final int cores = Math.max(1, Runtime.getRuntime().availableProcessors());

            // 1) Attempt to add QNN (Qualcomm Neural Network) if available
            /* Need to build from github with support - check chatGPT 
            ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: This binary was not compiled with ArmNN support.
            or 
            ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: QNN execution provider is not supported in this build. 

                Map<String, String> qnnOptions = new HashMap<>();
                // For example: qnnOptions.put("backend", "DSP");
                // qnnOptions.put("profiling_level", "default");
                // qnnOptions.put("remote_heap", "true"); etc.
                options.addQnn(qnnOptions);
                Log.i(TAG, "QNN EP added successfully.");
            } catch (Exception e) {
                Log.w(TAG, " or failed to load.", e);
            }
            */
            // 2) Attempt to add NNAPI
            try {
                options.addNnapi(); 
                // or options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED)) for specific flags
                Log.i(TAG, "NNAPI EP added successfully.");
            } catch (Exception e) {
                Log.w(TAG, "NNAPI EP not available or failed to load.", e);
            }

            // Add CPU option
            options.addCPU(true);

            // 5) Attempt to add XNNPACK
            try {
                Map<String, String> xnnpackOpts = Collections.emptyMap(); // or new HashMap<>() with any specific options
                options.addXnnpack(xnnpackOpts);
                Log.i(TAG, "XNNPACK EP added successfully.");
            } catch (Exception e) {
                Log.w(TAG, "XNNPACK EP not available or failed to load.", e);
            }
            
            // 2) Use a lower graph optimization level if battery is a bigger concern than speed
            // Start with ORT_ENABLE_ALL
            // Than ORT_ENABLE_EXTENDED or ORT_ENABLE_BASIC
            //options.setGraphOptimizationLevel(OrtSession.SessionOptions.OptLevel.ORT_ENABLE_EXTENDED);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            Log.i(TAG, "NNAPI available. Using hardware acceleration + ORT_ENABLE_ALL.");

            // 3) Limit threads (less CPU usage => lower battery drain)
            options.setIntraOpNumThreads(cores);     // use all cores
            options.setInterOpNumThreads(cores);     // allow parallel node exec
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

            //  CPU Arena Allocator: 
            //    - "true" can speed up repeated inferences by reusing memory blocks (often beneficial overall).
            //    - "false" might reduce memory overhead if you do few inferences. Usually "true" is recommended.
            options.setCPUArenaAllocator(true);

            // Memory Pattern Optimization:
            //    - "true" pre-allocates memory patterns for known shapes. Tends to speed up repeated inferences 
            //      => potentially less CPU time => less battery usage.
            //    - "false" if shapes vary wildly or you have memory constraints. 
            options.setMemoryPatternOptimization(true);

            options.setSessionLogVerbosityLevel(0);

            session = env.createSession(modelPath, options);
            inputNames = new ArrayList<>(session.getInputNames());

            hBuffer = FloatBuffer.wrap(h);
            hTensor = OnnxTensor.createTensor(env, hBuffer, h_tensorShape);
            
            //hTensor = OnnxTensor.createTensor(env, hBuffer, h_tensorShape);
            if (hTensor == null) {
                Log.e(TAG, "Failed to create hTensor.");
            }
            if (!Arrays.equals(hTensor.getInfo().getShape(), h_tensorShape)) {
                Log.e(TAG, "hTensor shape mismatch: Actual: " + Arrays.toString(hTensor.getInfo().getShape()) 
                        + ", Expected: " + Arrays.toString(h_tensorShape));
            }
            
            cBuffer = FloatBuffer.wrap(c);
            cTensor = OnnxTensor.createTensor(env, cBuffer, c_tensorShape);

            srTensor = OnnxTensor.createTensor(env, srBuffer, srShape);
            inputBuffer = FloatBuffer.allocate(Constants.FRAME_LENGTH);

            Log.d(TAG, "VADModelWrapper: Model loaded successfully from " + modelPath);
        } catch (OrtException e) {
            Log.d(TAG, "VADModelWrapper: Failed to load model from " + modelPath);
            e.printStackTrace();
            throw e;
        }
    }

    public int getFrameLength()
    {
        return Constants.FRAME_LENGTH;
    }

    private String copyAssetToInternalStorage(String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        File parentFile = file.getParentFile();
        if (!parentFile.exists()) {
            boolean dirsCreated = parentFile.mkdirs();
            Log.d(TAG, "Directories created: " + dirsCreated);
        }

        // Delete the existing file if it exists
        if (file.exists()) {
            boolean deleted = file.delete();
            Log.d(TAG, "Existing asset deleted: " + deleted);
        }

        if (!file.exists()) {
            try {
                AssetManager assetManager = context.getAssets();
                Log.d(TAG, "Attempting to open asset: " + assetName);
                InputStream in = assetManager.open(assetName);
                FileOutputStream out = new FileOutputStream(file);
                byte[] buffer = new byte[1024];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    out.write(buffer, 0, read);
                }
                in.close();
                out.close();
                Log.d(TAG, "Asset copied to internal storage: " + file.getAbsolutePath());
            } catch (IOException e) {
                Log.e(TAG, "Failed to copy asset file: " + assetName, e);
                return null;
            }
        
        } else {
            Log.d(TAG, "Asset already exists in internal storage: " + file.getAbsolutePath());
        }
        return file.getAbsolutePath();
    }
/*
    public List<float[]> splitAudioData(short[] x, int frameSize) {
        // Print the initial shape of x
        //System.out.println("Initial audio data length: " + x.length);
        //System.out.println("Frame size: " + frameSize);

        //for (int i = 0; i < chunks.size(); i++) {
        //    chunks.set(i, null); // Nullify the reference to each chunk
        //}
        chunks.clear(); // Clear the list after nullifying the chunks

        // Iterate over the audio data in steps of frameSize
        for (int i = 0; i < x.length; i += frameSize) {
            // Get the current chunk
            int end = Math.min(i + frameSize, x.length);
            short[] chunk = Arrays.copyOfRange(x, i, end);

            // Print the size of the current chunk
            // System.out.println("Chunk " + (i / frameSize) + " size before processing: " + chunk.length);

            // Check if the chunk size is equal to frameSize
            if (chunk.length == frameSize) {
                // Normalize the chunk and convert to float32
                for (int j = 0; j < chunk.length; j++) {
                    normalizedChunk[j] = chunk[j] / 32767.0f;
                }

                // Print the size of the chunk after processing
                //Log.d(TAG, "Chunk " + (i / frameSize) + " size after processing: " + normalizedChunk.length);
                //Log.d(TAG,  "Chunk " + (i / frameSize) + " size after processing: " + normalizedChunk.length);

                // Add the processed chunk to the list
                chunks.add(normalizedChunk);
            }
        }

        // Print the number of chunks created
        //Log.d(TAG, "Total number of chunks: " + chunks.size());
        return chunks;
    }*/
/*
public float predict_3(short[] x, int frameSize) throws OrtException {
    Deque<Float> framePredictions = new ArrayDeque<>();
    int numFrames = x.length / frameSize;
//    Log.d(TAG, "VADModelWrapper x.length: " + x.length);
//    Log.d(TAG, "VADModelWrapper frameSize: " + frameSize);
//    Log.d(TAG, "VADModelWrapper numFrames: " + numFrames);

    List<float[]> chunks = splitAudioData(x, frameSize);

//    List<Float> framePredictions = new ArrayList<>();

    int i = 0;
    for (float[] chunk : chunks) {
        chunk = Arrays.copyOf(chunk, chunk.length); // Equivalent of chunk.squeeze() in Python
        float[][] expandedChunk = {chunk}; // Equivalent of np.expand_dims(chunk, axis=0)

        Log.d(TAG, "VADModelWrapper chunk number: " + i);
        Log.d(TAG, "VADModelWrapper chunk: " + chunk);
        i++;

        long[] chunkShape = new long[]{1, chunk.length};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(expandedChunk[0]), chunkShape);
        OnnxTensor srTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{16000}), new long[]{1});
        OnnxTensor hTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(h), new long[]{2, 1, 64});
        OnnxTensor cTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(c), new long[]{2, 1, 64});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        List<String> inputNames = new ArrayList<>(session.getInputNames());
        inputs.put(inputNames.get(0), inputTensor);
        inputs.put(inputNames.get(1), srTensor);
        inputs.put(inputNames.get(2), hTensor);
        inputs.put(inputNames.get(3), cTensor);
        OrtSession.Result result = session.run(inputs);

        OnnxValue out = result.get(0);
        OnnxTensor hOutput = (OnnxTensor) result.get(1);
        OnnxTensor cOutput = (OnnxTensor) result.get(2);
        h = hOutput.getFloatBuffer().array();
        c = cOutput.getFloatBuffer().array();
        float[] outputData = ((OnnxTensor) out).getFloatBuffer().array();
        // framePredictions.add(outputData[0]);
        if (predictionBuffer.size() >= BUFFER_SIZE) {
            predictionBuffer.poll();
        }
        predictionBuffer.add(outputData[0]);
    }
    float sum = 0;
//    for (float prediction : framePredictions) {
    for (float prediction : predictionBuffer) {
        sum += prediction;
    }
    float meanPrediction = sum / predictionBuffer.size();
//    float meanPrediction = sum / framePredictions.size();
    return meanPrediction;
}
*/
public float predict_2(short[] x, int frameSize) throws OrtException {

    for (int i = 0; i < x.length; i++) {
        floatInput[i] = x[i] / 32767.0f;
    }
    //Log.d(TAG, "VADModelWrapper chunk number: " + i);
    //Log.d(TAG, "VADModelWrapper chunk: " + chunk);

    chunkShape[1] = floatInput.length;
    if (chunkShape[1] <= 0)
    {
        Log.d(TAG,  "chunkShape[1] <= 0 null");
        return 0;
    }
    //Log.d(TAG, "com.SdkPkh.Rlo chunkShape dimensions: [" + chunkShape[0] + ", " + chunkShape[1] + "]");
    inputBuffer.clear();
    inputBuffer.put(floatInput).flip();
    OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, chunkShape);
    
    hBuffer.clear(); // Reset position for writing
    hBuffer.put(h);  // Write updated values into the buffer
    hBuffer.flip();  // Prepare buffer for reading by the ONNX model

    // Update the buffer contents without reallocating
    cBuffer.clear(); // Reset position for writing
    cBuffer.put(c);  // Write updated values into the buffer
    cBuffer.flip();  // Prepare buffer for reading by the ONNX model

    // Update the buffer contents without reallocating
    srBuffer.clear(); // Reset position for writing
    srBuffer.put(srArray);  // Write updated values into the buffer
    srBuffer.flip();  // Prepare buffer for reading by the ONNX model

//        hTensor = OnnxTensor.createTensor(env, hBuffer, h_tensorShape);
//        cTensor = OnnxTensor.createTensor(env, cBuffer, c_tensorShape);
//        srTensor = OnnxTensor.createTensor(env, srBuffer, srShape);

    inputs.put("input", inputTensor);
    inputs.put("sr", srTensor);
    inputs.put("h", hTensor);
    inputs.put("c", cTensor);
    OrtSession.Result result = session.run(inputs);

    OnnxValue out = result.get(0);
    OnnxTensor hOutput = (OnnxTensor) result.get(1);
    OnnxTensor cOutput = (OnnxTensor) result.get(2);

    h = hOutput.getFloatBuffer().array();
    c = cOutput.getFloatBuffer().array();
    // Cast the result of getValue() to float[]
    //h = (float[]) hOutput.getValue();
    //c = (float[]) cOutput.getValue();
    float[] outputData = ((OnnxTensor) out).getFloatBuffer().array();
    return outputData[0];
}
    private float[] preprocessAudio(float[] audioData) {
        for (int i = 0; i < audioData.length; i++) {
            audioData[i] /= 32768.0f; // Normalize to [-1, 1]
        }
        return audioData;
    }

    // Method to print all accessible files starting from a given directory
    public void printAllFiles(File directory) {
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        printAllFiles(file); // Recursive call for directories
                    } else {
                        Log.d(TAG, "File: " + file.getAbsolutePath());
                    }
                }
            }
        } else {
            Log.d(TAG, "Directory does not exist: " + directory.getAbsolutePath());
        }
    }

        // Method to print all files in the assets directory
    public void printAllAssets(String path) {
        try {
            AssetManager assetManager = context.getAssets();
            String[] files = assetManager.list(path);
            if (files != null) {
                for (String file : files) {
                    String filePath = path.isEmpty() ? file : path + "/" + file;
                    if (assetManager.list(filePath).length > 0) {
                        Log.d(TAG, "Directory: " + filePath);
                        printAllAssets(filePath); // Recursive call for directories
                    } else {
                        Log.d(TAG, "Asset: " + filePath);
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to list assets in path: " + path, e);
        }
    }

        // Convenience method to print all files in the internal storage directory
    public void printAllFilesInInternalStorage() {
        printAllFiles(context.getFilesDir());
        printAllAssets("");
    }

}
