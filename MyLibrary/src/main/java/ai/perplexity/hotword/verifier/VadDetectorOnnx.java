package ai.perplexity.hotword.verifier;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import android.content.Context;

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
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;


public class VadDetectorOnnx {
    private final OrtSession session;
    private float[][][] h;
    private float[][][] c;
    private int lastSr = 0;
    private final long[] srArray = new long[]{16000};
    private final Map<String, OnnxTensor> inputs;
    private final OnnxTensor srTensor;
    private int lastBatchSize = 0;
    private static final List<Integer> SAMPLE_RATES = Arrays.asList(8000, 16000);
    private String TAG = "KeyWordsDetection VadDetectorOnnx";
    private Context context;
    private static final String[] QNN_LIB_FILENAMES = {
        "libQnnDsp.so",
        "libQnnCpu.so",
        "libQnnGpu.so",
        "libQnnHtp.so",
        "libQnnHta.so",
        // ... plus any you want, like libarm_compute.so, libonnxruntime.so, etc.
        // You can include "libarm_compute.so", "libonnxruntime.so" if also needed from assets
    };

    public static void printNativeLibraries(Context context) {
        // Get the native library directory
        String nativeLibPath = context.getApplicationInfo().nativeLibraryDir;
        //File libDir = new File(nativeLibPath);
        File libDir = new File(nativeLibPath).getParentFile();

        if (libDir.exists() && libDir.isDirectory()) {
            File[] files = libDir.listFiles();
            if (files != null && files.length > 0) {
                Log.i("NativeLibInspector", "Native libraries in jniLibs:");
                for (File file : files) {
                    Log.i("NativeLibInspector", " - " + file.getName());
                }
            } else {
                Log.i("NativeLibInspector", "No native libraries found in " + libDir.getAbsolutePath());
            }
        } else {
            Log.e("NativeLibInspector", "Native library directory not found: " + libDir.getAbsolutePath());
        }
    }

    private void copyAllQnnLibrariesFromAssets() {
        for (String libName : QNN_LIB_FILENAMES) {
            // String assetPath = "lib/arm64-v8a/" + libName; 
            String assetPath = libName; 
            String copiedPath = copyAssetToInternalStorage(assetPath, libName);

            if (copiedPath != null) {
                // Attempt to load the library. 
                // This ensures it's available in the process so that QNN can find symbols internally.
                try {
                    System.load(copiedPath);
                    Log.i(TAG, "System.load OK for " + copiedPath);
                } catch (UnsatisfiedLinkError ule) {
                    Log.e(TAG, "System.load failed for " + copiedPath, ule);
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // The same copyAssetToInternalStorage logic, but flexible for .so files
    // --------------------------------------------------------------------
    private String copyAssetToInternalStorage(String assetPath, String outputFileName) {
        File outFile = new File(context.getFilesDir(), outputFileName);
        if (outFile.exists()) {
            outFile.delete();
        }

        try (InputStream in = context.getAssets().open(assetPath);
            FileOutputStream out = new FileOutputStream(outFile)) {

            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            out.flush();
        } catch (IOException e) {
            Log.e(TAG, "Error copying asset " + assetPath + " to internal storage", e);
            return null;
        }

        Log.d(TAG, "Copied asset: " + assetPath + " -> " + outFile.getAbsolutePath());
        return outFile.getAbsolutePath();
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

    public VadDetectorOnnx(Context context, String modelPath) throws OrtException {
        this.context = context;
        // THIS IS WHERE THE MEMORY OVERRUN !!!!!!!!!!!!!!!!!!
        // If we return here all works well if not Mem overrun
        // RETURN ************
        // CHECK WITH LARGET STACK SIZE!!!
        //modelPath = copyAssetToInternalStorage(modelPath);
        printNativeLibraries(this.context);
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        // We’ll store the device’s nativeLib dir (where the .so files from jniLibs go)
        //String libDir = context.getApplicationInfo().nativeLibraryDir;
        String packageLibDir = context.getPackageResourcePath(); // Retrieve the package path
        File packageLib = new File(packageLibDir, "lib");

        boolean nnapiAdded = true;
        
        // 1) Attempt to add QNN (Qualcomm Neural Network) if available
        /* Need to build from github with support - check chatGPT 
        ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: This binary was not compiled with ArmNN support.
        or 
        ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: QNN execution provider is not supported in this build. 
*/
        // ----------------------------------------------------------------------
        //   QNN: Try adding multiple backends. Each attempt is in its own
        //   try/catch so a failure doesn't block other backends. 
        //   The "backend_path" must point to the relevant .so in jniLibs.
        // ----------------------------------------------------------------------
/*
        // (B) QNN DSP
        try {
            Map<String, String> qnnOptionsDSP = new HashMap<>();
            String libDir = copyAssetToInternalStorage("libQnnDsp.so");

            qnnOptionsDSP.put("backend_path", libDir);
            qnnOptionsDSP.put("backend", "DSP");
            options.addQnn(qnnOptionsDSP);
            Log.i(TAG, "QNN EP [DSP] added successfully (with backend_path).");
        } catch (Exception e) {
            Log.w(TAG, "QNN EP [DSP] not available or failed to load.", e);
        }
*//*
        // (D) QNN HTP (sometimes also referred to as HPC)
        try {
            Map<String, String> qnnOptionsHTP = new HashMap<>();
            qnnOptionsHTP.put("backend_path", libDir + "/libQnnHtp.so");
            qnnOptionsHTP.put("backend", "HTP");
            options.addQnn(qnnOptionsHTP);
            Log.i(TAG, "QNN EP [HTP] added successfully (with backend_path).");
        } catch (Exception e) {
            Log.w(TAG, "QNN EP [HTP] not available or failed to load.", e);
        }


        // (E) QNN HTA 
        try {
            Map<String, String> qnnOptionsHTA = new HashMap<>();
            qnnOptionsHTA.put("backend_path", libDir + "/libQnnHta.so");
            qnnOptionsHTA.put("backend", "HTA");
            options.addQnn(qnnOptionsHTA);
            Log.i(TAG, "QNN EP [HTA] added successfully (with backend_path).");
        } catch (Exception e) {
            Log.w(TAG, "QNN EP [HTA] not available or failed to load.", e);
        }

        // (C) QNN GPU
        try {
            Map<String, String> qnnOptionsGPU = new HashMap<>();
            qnnOptionsGPU.put("backend_path", libDir + "/libQnnGpu.so");
            qnnOptionsGPU.put("backend", "GPU");
            options.addQnn(qnnOptionsGPU);
            Log.i(TAG, "QNN EP [GPU] added successfully (with backend_path).");
        } catch (Exception e) {
            Log.w(TAG, "QNN EP [GPU] not available or failed to load.", e);
        }

        // (A) QNN CPU
        try {
            Map<String, String> qnnOptionsCPU = new HashMap<>();
            qnnOptionsCPU.put("backend_path", libDir + "/libQnnCpu.so");
            qnnOptionsCPU.put("backend", "CPU");  
            // Additional QNN parameters as desired:
            // qnnOptionsCPU.put("profiling_level", "default");
            // qnnOptionsCPU.put("remote_heap", "true");
            // ...
            options.addQnn(qnnOptionsCPU);
            Log.i(TAG, "QNN EP [CPU] added successfully (with backend_path).");
        } catch (Exception e) {
            Log.w(TAG, "QNN EP [CPU] not available or failed to load.", e);
        }
*/
/*
        try {
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
        /* Need to build from github with support - check chatGPT 
        ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: This binary was not compiled with ArmNN support.
        or 
        ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT - message: QNN execution provider is not supported in this build. 

        // 3) Attempt to add ArmNN
        try {
            boolean useArena = true;  // or false
            options.addArmNN(useArena);
            Log.i(TAG, "ArmNN EP added successfully.");
        } catch (Exception e) {
            Log.w(TAG, "ArmNN EP not available or failed to load.", e);
        }
        */
        // 4) Attempt to add ACL (ARM Compute Library)
        try {
            boolean enableFastMath = true;  // or false
            options.addACL(enableFastMath);
            Log.i(TAG, "ACL EP added successfully.");
        } catch (Exception e) {
            Log.w(TAG, "ACL EP not available or failed to load.", e);
        }

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
        if (nnapiAdded) {
            // The device supports NNAPI. 
            // Possibly keep the highest graph optimization level (ORT_ENABLE_ALL) 
            // and single-thread or multi-thread settings as desired.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            Log.i(TAG, "NNAPI available. Using hardware acceleration + ORT_ENABLE_ALL.");
        } else {
            // Fallback if no NNAPI driver. 
            // You might choose fewer threads, different optimization, etc.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT);
            Log.i(TAG, "NNAPI not available, using CPU fallback + ORT_ENABLE_EXTENDED.");
        }

        // 3) Limit threads (less CPU usage => lower battery drain)
        options.setIntraOpNumThreads(1);
        options.setInterOpNumThreads(1);

        // 4) Optionally set execution mode
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

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
        resetStates();
        // Create srTensor once with srArray
        srTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(srArray), new long[]{1});
        inputs = new HashMap<>();
        inputs.put("sr", srTensor);
    }

    void resetStates() {
        h = new float[2][1][64];
        c = new float[2][1][64];
        lastSr = 0;
        lastBatchSize = 0;
    }

    public void close() throws OrtException {
        session.close();
    }

    public static class ValidationResult {
        public final float[][] x;
        public final int sr;

        public ValidationResult(float[][] x, int sr) {
            this.x = x;
            this.sr = sr;
        }
    }

    private ValidationResult validateInput(float[][] x, int sr) {
        if (x.length == 1) {

            x = new float[][]{x[0]};
/// Always happens            throw new IllegalArgumentException("x.length == 1");
        }
        return new ValidationResult(x, sr);
    }

    public float[] call(float[][] x, int sr) throws OrtException {
        int batchSize = x.length;

        if (lastBatchSize == 0 || lastSr != sr || lastBatchSize != batchSize) {
            System.out.println("RESET STATES ??????????????????????");
            resetStates();
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OnnxTensor inputTensor = null;
        OnnxTensor hTensor = null;
        OnnxTensor cTensor = null;
        OrtSession.Result ortOutputs = null;

        try {
            inputTensor = OnnxTensor.createTensor(env, x);
            hTensor = OnnxTensor.createTensor(env, h);
            cTensor = OnnxTensor.createTensor(env, c);
            //srTensor = OnnxTensor.createTensor(env, new long[]{sr});

            //Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input", inputTensor);
            //inputs.put("sr", srTensor);
            inputs.put("h", hTensor);
            inputs.put("c", cTensor);

            ortOutputs = session.run(inputs);
            float[][] output = (float[][]) ortOutputs.get(0).getValue();
            h = (float[][][]) ortOutputs.get(1).getValue();
            c = (float[][][]) ortOutputs.get(2).getValue();

            lastSr = sr;
            lastBatchSize = batchSize;
            return output[0];
        } finally {
            inputs.remove("input");
            inputs.remove("h");
            inputs.remove("c");
            if (inputTensor != null) {
                inputTensor.close();
            }
            if (hTensor != null) {
                hTensor.close();
            }
            if (cTensor != null) {
                cTensor.close();
            }
            if (ortOutputs != null) {
                ortOutputs.close();
            }
        }
    }
}
