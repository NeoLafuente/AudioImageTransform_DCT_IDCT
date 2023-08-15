# Audio-to-Image and Image-to-Audio Transformation - DCT | IDCT

This code demonstrates a process of transforming audio signals into images using the Discrete Cosine Transform (DCT) and then reconstructing the audio signals from the images using the Inverse DCT (IDCT). This technique is useful for understanding the transformation between audio and image domains, as well as exploring potential signal processing and compression applications.

**Note:** The provided input audio file "WhatsLoveGottoDowithIt_TinaTurner_Mono.wav" cannot be included due to copyright restrictions.

## Requirements

Before running the code, make sure you have the following libraries installed:

- numpy
- matplotlib
- scipy
- cupy
- PIL
- imageio

## Steps

1. **Audio to Frequency Array (DCT Domain):** The input audio signal is read from a WAV file. It is divided into segments, and a DCT is applied to each segment. The signs of the DCT matrix are extracted, and the magnitude of the DCT coefficients is computed and log-scaled to create a frequency array image.

2. **Frequency Array Image Visualization:** The frequency array image is displayed and saved. It provides a visual representation of the audio signal in the frequency domain.

3. **Frequency Array Image to Reconstructed Audio:** The frequency array image is normalized and converted back to a frequency array using the inverse scaling. The inverse DCT is applied to each segment of the frequency array to reconstruct the audio signal.

4. **Audio Comparisons:** The original audio signal is compared with the reconstructed audio signal using correlation and squared error metrics. The closer the correlation is to 1 and the lower the squared error, the better the reconstruction.

5. **Frequency Filtering and Correlation Analysis:** Different frequency ranges of the reconstructed audio signal are analyzed by applying bandpass filters. The correlation between the original and reconstructed signals for each frequency range is computed to understand information loss.

6. **Saving Frequency Array as Image:** The frequency array image is saved as an image file for further analysis or visualization.

7. **Image to Reconstructed Audio:** The saved frequency array image is read and converted back to the frequency array. The inverse DCT is applied to reconstruct the audio signal from the frequency array.

8. **Audio Comparisons (Image-based Reconstruction):** The correlation between the original audio and the reconstructed audio from the image is calculated. Additionally, the reconstructed audio signal is filtered to the human audible range (20 Hz - 20 KHz), and correlation is computed.

9. **Saving Filtered Reconstructed Audio:** The filtered reconstructed audio is saved as a WAV file for further analysis or playback.

## Results

The following correlations were calculated for different scenarios:

- Original signal & reconstruction: Correlation (audio-to-audio)
- Original signal & reconstruction (20Hz-20KHz): Correlation after filtering to audible range
- Original signal & reconstruction from image: Correlation after image-to-audio reconstruction
- Original signal & reconstruction from image (20Hz-20KHz): Correlation after image-to-audio reconstruction and filtering

## Case Correlation
- original signal & reconstruction 0.9996507156192037
- original signal & reconstruction (20Hz-20KHz) 0.9996424377971151
- original signal & reconstruction from image 0.9970420387637196
- original signal & reconstruction from image (20Hz-20KHz) 0.9968352474179451


These correlations demonstrate the effectiveness of the audio-to-image and image-to-audio transformations using the DCT | IDCT process. The technique shows strong correlation and accuracy in preserving audio information, even after image domain transformation.

Remember that this code serves as a demonstrative example of audio and image transformations and their effects on audio signal reconstruction. It provides insights into the potential applications and limitations of these transformations in signal processing and compression contexts.
