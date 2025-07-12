# Audio Visualizer and Analyzer

This project is a real-time audio visualizer that processes WAV audio files, analyzes their frequency spectrum using the Fast Fourier Transform (FFT), and displays the results using SDL2.

![Visualizer Screenshot](./assets/GUI.png)

## Screenshots

Below are a couple of screenshots showcasing the application in action.

| Main GUI                                   | Active Note Detection (Terminal)            |
| ------------------------------------------| ------------------------------------------- |
| ![Visualizer GUI](./assets/GUI.png)       | ![Terminal Output](./assets/active_notes_terminal.png) |

## How It Works

The application processes audio in a series of sequential steps to produce real-time visualization:

1. **WAV File Loading**  
   The program starts by loading a user-specified `.wav` file. It reads the header into a `WavHeader` struct and validates the format. The raw audio data is then normalized into a `std::vector<double>` of samples (ranging from -1.0 to 1.0) and stored in a `WavFile` struct.

2. **Audio Playback**  
   SDL2's audio API is used to stream the audio asynchronously. Playback is managed through a callback function, with controls for pausing and resuming based on the stream state.

3. **Frame Processing**  
   The audio is divided into overlapping frames for visualization. Each frame undergoes:
   - A **Hanning Window** application to reduce spectral leakage.
   - **Fast Fourier Transform (FFT)** to convert samples from the time domain to the frequency domain.

4. **Frequency Analysis & Note Detection**  
   - The FFT output is converted to decibel scale, and an **A-weighting** filter is applied to adjust for human hearing sensitivity.
   - **Peak Detection** identifies significant peaks above a noise threshold.
   - For each peak, a **Binary Search** is performed on a pre-computed note table to find the closest musical note within a set tolerance.

5. **Rendering and Visualization**  
   - The frequency spectrum is displayed as logarithmically-scaled bars. Magnitudes are smoothed using previous frame data stored in a `std::deque`, creating fluid animation.
   - Detected musical notes are displayed with a persistence effect to avoid flickering.
   - Rendering is handled via SDL2 and SDL2_ttf.

## Core Algorithms and Data Structures

- **Fast Fourier Transform (FFT)**  
  Implements a recursive, in-place **Cooley-Tukey** algorithm via the `fft` function, transforming a `std::vector<std::complex<double>>` of audio samples.

- **Binary Search for Note Detection**  
  A sorted `std::vector<MusicalNote>` allows efficient note identification using `std::lower_bound` for logarithmic search time—crucial for real-time performance.

- **Hanning Window**  
  A basic loop applies the Hanning formula:  
  `0.5 * (1 - cos(2 * PI * i / (N - 1)))`  
  to each frame sample in `applyHannWindow`, reducing leakage in frequency analysis.

- **Data Structures**  
  - `WavHeader` / `WavFile`: C-style structs map the binary WAV header and store normalized samples.
  - `FrequencyBand`: Represents visual bars with frequency range, magnitude, and color.
  - `Note`: Manages the state of detected musical notes, with persistence tracking.
  - `AudioVisualizer` Class: Encapsulates SDL window/renderer, audio state, and core loop.
  - `std::deque<vector<double>>`: Stores recent magnitudes for smoothing animations.

## Project Structure

```
.
├── .gitignore
├── README.md
├── Makefile
├── bin/
│   ├── SDL2.dll
│   └── SDL2_ttf.dll
├── build/
├── assets/
│   ├── test.wav
│   └── piano.wav
├── include/
│   └── SDL2/
├── lib/
└── src/
    └── main.cpp
```

## Getting Started

### Prerequisites

- A C++ compiler (e.g., G++)
- `make`
- SDL2 and SDL2_ttf (included in this repository)

### Installation & Compilation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repository-folder>
   ```

2. **Compile the project:**
   Use the provided Makefile:
   ```bash
   make
   ```
   The executable will be created in the `build/` directory.

### Running the Visualizer

Run the executable from the project root, passing a path to a WAV file. Ensure the `bin/` directory is present so the required `.dll` files can be located.

```bash
./build/main.exe assets/test.wav
```

### Controls

- `SPACE`: Pause/Resume playback
- `I`: Toggle isolation mode (highlight active bands)
- `ESC`: Exit the application

## Contributions

- **Yugam Bhatt (2023aib1020)**
  - Note detection and peak analysis
  - Magnitude/decibel conversion
  - Spectrum rendering with 60Hz smoothing
  - Error handling

- **Vaibhav Gupta (2023aib1019)**
  - Cooley-Tukey FFT implementation
  - Frequency band generation
  - Project ideation and research

- **Shashwat Saini (2023aib1015)**
  - Hanning window implementation
  - A-weighting and frequency scaling
  - SDL2 graphics setup and color mapping

- **Nitin Kumar (2023aib1012)**
  - WAV file parsing with metadata handling
  - SDL audio playback implementation
  - Audio buffer management
