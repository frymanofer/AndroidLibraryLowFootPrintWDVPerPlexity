# mkdir ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example/android/app/libs
# mkdir ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example_car_parking/android/app/libs
# rm -rf ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example_car_parking/android/app/libs/MyLibrary-release.aar
# rm -rf ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example/android/app/libs/MyLibrary-release.aar
# cp -a MyLibrary/build/outputs/aar/MyLibrary-release.aar ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example/android/app/libs/
# cp -a MyLibrary/build/outputs/aar/MyLibrary-release.aar ../VAD_WAKEWORD_FILES/wakewords/ReactNative_WakeWordDetection/example_car_parking/android/app/libs/
# cp MyLibrary/build/outputs/aar/MyLibrary-release.aar ../porcuSafe/android/app/libs/MyLibrary-release.aar

cp MyLibrary/build/outputs/aar/MyLibrary-release.aar ../android-hotword/android_perplexity/libs/ai/perplexity/hotword/1.0.0/keyworddetection-1.0.0.aar