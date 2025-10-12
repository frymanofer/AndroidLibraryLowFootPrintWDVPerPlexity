# Keep everything in your package
-keep class ai.perplexity.hotword.verifier.** { *; }
-keep class com.davoice.speakerid.** { *; }     # <-- MISSING; add this

# Keep everything in the ONNX Runtime package
-keep class ai.onnxruntime.** { *; }

# Keep exceptions
-keepnames class ai.onnxruntime.OrtException

# Keep JNI-bound methods
-keepclassmembers class * {
    native <methods>;
}

# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }

# Keep AndroidX Lifecycle classes
-keep class androidx.lifecycle.** { *; }

# Keep AndroidX AppCompat classes
-keep class androidx.appcompat.** { *; }

# Multidex support
-keep class androidx.multidex.** { *; }

# Keep React Native classes (if used)
-keep class com.facebook.react.** { *; }

# Keep all classes that have JNI bindings (used by ONNX Runtime or custom libraries)
-keepclassmembers class * {
    native <methods>;
}

# Keep JNI libraries in APK
-keepclasseswithmembers class * {
    static void loadLibrary(java.lang.String);
}

# Ensure native libraries are not removed during optimization
-keepnames class * {
    native <methods>;
}

# Do not shrink or obfuscate methods used via reflection (e.g., JNI calls)
-keepattributes *Annotation*
-keepattributes EnclosingMethod
-keepattributes InnerClasses
-keepattributes Signature

# Retain the ONNX Runtime JNI loader
-keepclassmembers class ai.onnxruntime.OrtEnvironment {
    static <methods>;
}
