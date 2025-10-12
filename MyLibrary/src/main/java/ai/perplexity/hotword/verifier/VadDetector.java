package ai.perplexity.hotword.verifier;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import android.content.Context;

import ai.onnxruntime.OrtException;

public class VadDetector {
    private final VadDetectorOnnx model;
    private String TAG = "KeyWordsDetection VadDetector";
    float[] audioData;
    private final float[][] preAllocatedInput = new float[1][];

    public VadDetector(Context context, String modelPath
                                ) throws OrtException {

        this.model = new VadDetectorOnnx(context, modelPath);
        audioData = new float[Constants.FRAME_LENGTH];
        reset();
    }

    public void reset() {
        model.resetStates();
    }
    
//    public float predict_2(byte[] data) {
    public float predict_2(short[] data) {
    /*    float[] audioData = new float[data.length / 2];
        for (int i = 0; i < audioData.length; i++) {
            audioData[i] = ((data[i * 2] & 0xff) | (data[i * 2 + 1] << 8)) / 32767.0f;
        }*/
        if (data.length != audioData.length) {
            System.out.println("data.legth != Constants.FRAME_LENGTH");
            System.out.println("data.legth == " + data.length);
            System.out.println("audioData.lenght == " + audioData.length);
            System.out.println("Constants.FRAME_LENGTH == " + Constants.FRAME_LENGTH);
            System.out.println("Reallocating audioDate == " + Constants.FRAME_LENGTH);
            audioData = new float[data.length];
        }

        for (int i = 0; i < audioData.length; i++) {
            audioData[i] = data[i] / 32767.0f;
        }
        preAllocatedInput[0] = audioData;

        float speechProb = 0.0f;
        try {
            speechProb = model.call(preAllocatedInput, Constants.SAMPLE_RATE)[0];
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return speechProb;
    }

    public void close() throws OrtException {
        reset();
        model.close();
    }
}
