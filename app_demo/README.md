# Facer

This app is only used for acoustic sensing data collection. The full-stack facial expression recognition function is to be done in the future.

## Building

The application can be built using Android Studio and Gradle. If you have your own compiled LibAS library (```.aar```), you can put it under```libacousticsensing``` folder and replace the ```.aar``` file with the same filename.

## Acknowledgement

The Facer data collection app is built based on (Chaperone)[https://github.com/cryspuwaterloo/chaperone].

Chaperone is a standalone acoustic sensing app developed based on LibAcousticSensing (LibAS)[https://github.com/yctung/LibAcousticSensing]. We use a pre-compiled library under the ```libacousticsensing``` directory.

## Toubleshooting (Chaperone)

- Debug field shows "Init. fails" or Distance field never changes: It means it cannot find the preamble and conduct acoustic sensing at this moment. It usually happens when you manually start the acoustic sensing. Just try to restart it until it works. If it fails several times, please check if your device is supported and if the speaker and the microphone of your phone are available.

- Failed to start acoustic sensing: please make sure that the phone is not under "Do Not Disturb" mode and try again.
