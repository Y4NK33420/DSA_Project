#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>
#include <map>
#include <deque>
#include <string>
#include <algorithm>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <math.h>

using namespace std;

constexpr double PI = 3.14159265358979323846;
constexpr int barDiff = 200; // Frequency difference between bars in Hz
constexpr double minFreq = 20.0;    // Minimum frequency to analyze
constexpr double maxFreq = 20000.0; // Maximum frequency to analyze
constexpr int numBands = 32;        // Number of frequency bands
const double freqMultiplier = pow(maxFreq / minFreq, 1.0 / numBands);

constexpr double f1 = 20.598997; 
constexpr double f2 = 107.65265;
constexpr double f3 = 737.86223;
constexpr double f4 = 12194.217;

// Structure to hold WAV file header information
struct WavHeader {
    char riff[4];
    uint32_t fileSize;
    char wave[4];
    char fmt[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

struct FrequencySubBand {
    int startFreq;
    int endFreq;
    double magnitude;
    double smoothedMagnitude;
};

struct FrequencyBand {
    int startFreq;
    int endFreq;
    double magnitude;
    double smoothedMagnitude;
    SDL_Color color;
    string name;
    vector<FrequencySubBand> subBands;
};

struct WavFile {
    WavHeader header;
    vector<double> samples;
};

struct Note {
    string name;
    double frequency;
    double magnitude;
    bool active;
    int persistenceFrames;
};



struct MusicalNote {
    string name;
    double frequency;
    
    bool operator<(const MusicalNote& other) const {
        return frequency < other.frequency;
    }
};

// Sorted array of notes spanning multiple octaves
const vector<MusicalNote> NOTES = []() {
    vector<MusicalNote> notes;
    const vector<pair<string, double>> baseNotes = {
        {"C", 261.63}, {"C#", 277.18}, {"D", 293.66}, {"D#", 311.13},
        {"E", 329.63}, {"F", 349.23}, {"F#", 369.99}, {"G", 392.00},
        {"G#", 415.30}, {"A", 440.00}, {"A#", 466.16}, {"B", 493.88}
    };
    
    // Generate notes for octaves 0-8
    for (int octave = 0; octave <= 8; ++octave) {
        double octaveMultiplier = pow(2, octave - 4); // Center around octave 4
        for (const auto& [noteName, baseFreq] : baseNotes) {
            notes.push_back({
                noteName + to_string(octave),
                baseFreq * octaveMultiplier
            });
        }
    }
    return notes;
}();

string identifyNote(double frequency) {
    if (frequency <= 0) return "";
    
    const double TOLERANCE_PERCENT = 3.0; // 3% tolerance for frequency matching
    
    // Binary search for the closest note
    auto it = lower_bound(NOTES.begin(), NOTES.end(), MusicalNote{"", frequency},
        [](const MusicalNote& a, const MusicalNote& b) {
            return a.frequency < b.frequency;
        });
    
    // Handle edge cases
    if (it == NOTES.end()) return NOTES.back().name;
    if (it == NOTES.begin()) return NOTES.front().name;
    
    // Find the closest note by comparing distances
    auto prevIt = prev(it);
    double prevDiff = abs(frequency - prevIt->frequency);
    double currDiff = abs(frequency - it->frequency);
    
    // Calculate allowed tolerance ranges
    double prevTolerance = prevIt->frequency * (TOLERANCE_PERCENT / 100.0);
    double currTolerance = it->frequency * (TOLERANCE_PERCENT / 100.0);
    
    // If frequency is not within tolerance of either note, return empty string
    if (prevDiff > prevTolerance && currDiff > currTolerance) {
        return "";
    }
    
    return (prevDiff < currDiff) ? prevIt->name : it->name;
}

// Apply Hanning window to reduce spectral leakage
void applyHannWindow(vector<complex<double>>& samples) {
    int N = samples.size();
    for (int i = 0; i < N; i++) {
        double multiplier = 0.5 * (1 - cos(2 * PI * i / (N - 1)));
        samples[i] *= multiplier;
    }
}

// FFT implementation using Cooley-Tukey algorithm
void fft(vector<complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return;

    vector<complex<double>> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even);
    fft(odd);

    for (int k = 0; k < N / 2; ++k) {
        complex<double> t = polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// Load and validate WAV file
WavFile loadWav(const string& filename) {
    WavFile wav;
    ifstream file(filename, ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Failed to open WAV file.");
    }

    // Read WAV header
    file.read(reinterpret_cast<char*>(&wav.header), sizeof(WavHeader));

    // Validate WAV format
    if (string(wav.header.riff, 4) != "RIFF" ||
        string(wav.header.wave, 4) != "WAVE" ||
        string(wav.header.fmt, 4) != "fmt ") {
        throw runtime_error("Invalid WAV file format.");
    }

    // Find data chunk
    char chunkId[4];
    uint32_t chunkSize;
    while (true) {
        file.read(chunkId, 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);
        
        if (string(chunkId, 4) == "data") {
            break;
        }
        file.seekg(chunkSize, ios::cur);
    }

    // Read audio samples
    int numSamples = chunkSize / (wav.header.bitsPerSample / 8);
    wav.samples.resize(numSamples);

    if (wav.header.bitsPerSample == 16) {
        for (int i = 0; i < numSamples; ++i) {
            int16_t sample;
            file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
            wav.samples[i] = static_cast<double>(sample) / 32768.0;
        }
    } else if (wav.header.bitsPerSample == 24) {
        for (int i = 0; i < numSamples; ++i) {
            uint8_t bytes[3];
            file.read(reinterpret_cast<char*>(bytes), 3);
            int32_t sample = (bytes[2] << 16) | (bytes[1] << 8) | bytes[0];
            if (sample & 0x800000) sample |= 0xFF000000;
            wav.samples[i] = static_cast<double>(sample) / 8388608.0;
        }
    } else {
        throw runtime_error("Unsupported bit depth: " + to_string(wav.header.bitsPerSample));
    }

    file.close();
    return wav;
}

double calculateAWeighting(double f) {
    double f2_2 = f * f;
    double r = (f2_2 * f2_2) * f4 * f4;
    double s = (f2_2 + f1 * f1) * sqrt((f2_2 + f2 * f2) * (f2_2 + f3 * f3)) * (f2_2 + f4 * f4);
    double a = 1.2589 * r / s;  // 1.2589 = 10^(2.0/20)
    return a;
}

struct NotePeak {
    double frequency;
    double magnitude;
    string noteName;
};

// Helper function to convert FFT bin index to frequency
double binToFrequency(int bin, int sampleRate, int fftSize) {
    return static_cast<double>(bin) * sampleRate / fftSize;
}

vector<NotePeak> detectActivePeaks(const vector<double>& magnitudes, int sampleRate) {
    vector<NotePeak> peaks;
    const double PEAK_THRESHOLD = -30.0;  // dB threshold for peak detection
    
    for (size_t i = 1; i < magnitudes.size() - 1; i++) {
        if (magnitudes[i] > PEAK_THRESHOLD &&
            magnitudes[i] > magnitudes[i-1] &&
            magnitudes[i] > magnitudes[i+1]) {
            
            double freq = binToFrequency(i, sampleRate, magnitudes.size() * 2);
            string note = identifyNote(freq);
            
            if (!note.empty()) {
                peaks.push_back({freq, magnitudes[i], note});
            }
        }
    }
    return peaks;
}

// Enhanced visualization class
class AudioVisualizer {
private:
    bool isolationMode = false;
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    int windowWidth;
    int windowHeight;
    int windowSize;
    vector<FrequencyBand> bands;
    deque<vector<double>> history;
    vector<Note> activeNotes;
    
    static const int HISTORY_SIZE = 50;
    static const int MIN_MAGNITUDE_THRESHOLD = 0.1;

    SDL_AudioDeviceID audioDevice;
    const WavFile* currentWav;
    size_t audioPos;
    const WavFile* wav;
    
    static const int NOTE_PERSISTENCE_FRAMES = 10;  // value to control how long notes persist

    bool isPaused = false;
    SDL_mutex* audioMutex;  // For thread-safe audio processing

    // Audio callback function must be static
    static void audioCallback(void* userdata, Uint8* stream, int len) {
        AudioVisualizer* vis = static_cast<AudioVisualizer*>(userdata);
        vis->generateAudio(stream, len);
    }
    
    void generateAudio(Uint8* stream, int len) {
        SDL_LockMutex(audioMutex);
        
        if (isPaused) {
            // Output silence when paused
            SDL_memset(stream, 0, len);
        } else {
            int samplesPerCallback = len / sizeof(int16_t);
            int16_t* audio = reinterpret_cast<int16_t*>(stream);
            
            for (int i = 0; i < samplesPerCallback && audioPos < currentWav->samples.size(); i++) {
                audio[i] = static_cast<int16_t>(currentWav->samples[audioPos++] * 32767);
            }
            
            // Loop playback
            if (audioPos >= currentWav->samples.size()) {
                audioPos = 0;
            }
        }
        
        SDL_UnlockMutex(audioMutex);
    }

    void initAudio(const WavFile& wav) {
        SDL_AudioSpec want, have;
        SDL_zero(want);
        want.freq = wav.header.sampleRate;
        want.format = AUDIO_S16SYS;
        want.channels = wav.header.numChannels;
        want.samples = 4096;
        want.callback = audioCallback;
        want.userdata = this;

        audioDevice = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (audioDevice == 0) {
            throw runtime_error("Failed to open audio device: " + string(SDL_GetError()));
        }

        currentWav = &wav;
        audioPos = 0;
        SDL_PauseAudioDevice(audioDevice, 0); // Start playing
    }

    void initSDL() {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) {
            throw runtime_error("SDL initialization failed: " + string(SDL_GetError()));
        }
        
        if (TTF_Init() < 0) {
            throw runtime_error("SDL_ttf initialization failed: " + string(TTF_GetError()));
        }

        window = SDL_CreateWindow("Music Visualizer",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            windowWidth, windowHeight,
            SDL_WINDOW_SHOWN);
            
        if (!window) {
            throw runtime_error("Window creation failed: " + string(SDL_GetError()));
        }

        renderer = SDL_CreateRenderer(window, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            
        if (!renderer) {
            throw runtime_error("Renderer creation failed: " + string(SDL_GetError()));
        }

        font = TTF_OpenFont("C:\\Windows\\Fonts\\arial.ttf", 16);
    }

    void renderText(const string& text, int x, int y, SDL_Color color) {
        if (!font) {
            cout << "Font not loaded!" << endl;
            return;
        }
        
        SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), color);
        if (!surface) {
            cout << "Failed to create text surface: " << TTF_GetError() << endl;
            return;
        }
        
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        if (!texture) {
            cout << "Failed to create texture: " << SDL_GetError() << endl;
            SDL_FreeSurface(surface);
            return;
        }
        
        SDL_Rect rect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(renderer, texture, nullptr, &rect);
        SDL_DestroyTexture(texture);
        SDL_FreeSurface(surface);
    }

    void renderWaveform(const vector<double>& magnitudes, 
                       const WavFile& wav,
                       const vector<complex<double>>& windowedSamples) {
        int bandWidth = windowWidth / bands.size();
        int subBandWidth = bandWidth / 4;  // 4 subbands per band
        
        // Update history
        history.push_back(magnitudes);
        if (history.size() > HISTORY_SIZE) {
            history.pop_front();
        }

        // Render frequency bands with history
        for (size_t i = 0; i < bands.size(); ++i) {
            double& smoothedMag = bands[i].smoothedMagnitude;
            smoothedMag = smoothedMag * 0.7 + magnitudes[i] * 0.3;

            // Render subgroups
            for (size_t j = 0; j < bands[i].subBands.size(); ++j) {
                auto& subBand = bands[i].subBands[j];
                
                // Calculate subband magnitude based on frequency range
                double subMagnitude = 0.0;
                int count = 0;
                
                for (int k = 0; k < windowSize / 2; ++k) {
                    double frequency = (k * wav.header.sampleRate) / windowSize;
                    if (frequency >= subBand.startFreq && frequency < subBand.endFreq) {
                        subMagnitude += abs(windowedSamples[k]) / windowSize;
                        count++;
                    }
                }
                
                if (count > 0) {
                    subMagnitude /= count;
                }
                
                // Smooth the subband magnitude
                subBand.smoothedMagnitude = subBand.smoothedMagnitude * 0.8 + subMagnitude * 0.2;

                // Render the subband
                int xPos = i * bandWidth + j * subBandWidth;
                int height = static_cast<int>(subBand.smoothedMagnitude * windowHeight * 2);
                
                SDL_SetRenderDrawColor(renderer,
                    bands[i].color.r,
                    bands[i].color.g,
                    bands[i].color.b,
                    200);  // Slightly transparent
                    
                SDL_Rect bar = {
                    xPos,
                    windowHeight - height,
                    subBandWidth - 1,
                    height
                };
                SDL_RenderFillRect(renderer, &bar);
            }

            // Render band name
            if (i == 0 || i == bands.size() / 2 || i == bands.size() - 1) {
                SDL_Color textColor = {255, 255, 255, 255};
                renderText(bands[i].name,
                    i * bandWidth,
                    windowHeight - 20,
                    textColor);
            }
        }
    }

    void renderNotes() {
        SDL_Color textColor = {255, 255, 255, 255};
        string noteText = "Active Notes: ";
        
        bool hasNotes = false;
        for (const auto& note : activeNotes) {
            if (note.active) {
                cout << "Rendering note: " << note.name << " with magnitude: " << note.magnitude << endl;
            }
            
            if (note.active && note.magnitude > 0.0005) {  // Match the detection threshold
                if (hasNotes) noteText += ", ";
                noteText += note.name + "(" + to_string(note.magnitude).substr(0,6) + ")";
                hasNotes = true;
            }
        }
        
        if (!hasNotes) noteText += "None";
        cout << "Rendering text: " << noteText << endl; 
        renderText(noteText, 10, 10, textColor);
    }

    static constexpr int N = 32768; // FFT frame size
    int framePointer;
    vector<double> magnitudesDB; // Store magnitudes in 

    double toDecibels(double magnitude) {
        const double MIN_DB = -60.0;
        if (magnitude <= 0.0) return MIN_DB;
        double db = 20.0 * log10(magnitude);
        return max(db, MIN_DB);
    }

    void processFrame(const WavFile& wav, vector<complex<double>>& fftData) {
        // Clear the FFT data
        fill(fftData.begin(), fftData.end(), complex<double>(0.0, 0.0));

        // Copy samples and apply window
        for (int i = 0; i < N && (framePointer + i) < wav.samples.size(); ++i) {
            double hannWindow = 0.5 * (1.0 - cos(2.0 * PI * i / (N - 1)));
            double sample = wav.samples[framePointer + i] * hannWindow;
            fftData[i] = complex<double>(sample, 0.0);
        }

        // Perform FFT
        fft(fftData);

        // Calculate magnitudes with more aggressive scaling
        magnitudesDB.resize(N/2);
        for (int i = 0; i < N/2; ++i) {
            double magnitude = abs(fftData[i]);
            magnitude = magnitude * 100.0 / N; 
            magnitudesDB[i] = toDecibels(magnitude);
        }

        // Decrease persistence counters and clear inactive notes
        for (auto& note : activeNotes) {
            if (note.active) {
                if (note.persistenceFrames > 0) {
                    note.persistenceFrames--;
                }
                if (note.persistenceFrames == 0) {
                    note.active = false;
                    note.magnitude = 0.0;
                }
            }
        }

        // Detect notes from frequency spectrum
        for (int i = 0; i < N/2; i++) {
            double frequency = (i * wav.header.sampleRate) / static_cast<double>(N);
            double magnitude = abs(fftData[i]) / static_cast<double>(N);
            
            if (magnitude > 0.001) {  
                string noteName = identifyNote(frequency);
                if (!noteName.empty()) {
                    auto it = find_if(activeNotes.begin(), activeNotes.end(),
                        [&noteName](const Note& n) { return n.name == noteName; });
                    if (it != activeNotes.end()) {
                        cout << "Activating note: " << noteName 
                                  << " with magnitude: " << magnitude << endl;
                        it->active = true;
                        it->magnitude = max(it->magnitude, magnitude);
                        it->persistenceFrames = NOTE_PERSISTENCE_FRAMES;
                    } else {
                        cout << "Note found but not in activeNotes: " << noteName << endl;
                    }
                }
            }
        }

        cout << "\nCurrently active notes:" << endl;
        for (const auto& note : activeNotes) {
            if (note.active && note.persistenceFrames > 0) {
                cout << note.name << " (mag: " << note.magnitude 
                          << ", frames: " << note.persistenceFrames << ")" << endl;
            }
        }
    }

    void renderSpectrum() {
        const int numBands = N/64;  
        const int samplesPerBand = 8;  
        const int subBands = 2;  
        const int barWidth = max(4, windowWidth / (numBands * subBands));  
        
        const double maxDB = 0.0;
        const double minDB = -50.0;  
        
        const int labelHeight = 20;
        const int graphHeight = windowHeight - labelHeight;
        
        // Smoothing parameters
        static vector<double> smoothedHeights(numBands * subBands, 0.0);
        static vector<double> targetHeights(numBands * subBands, 0.0);
        const double attackTime = 0.1;   // Fast attack for transients
        const double releaseTime = 0.15; // Slower release for smoother falloff
        
        // Detect active peaks if in isolation mode
        vector<NotePeak> activePeaks;
        if (isolationMode) {
            activePeaks = detectActivePeaks(magnitudesDB, wav->header.sampleRate);
        }

        // Process main frequency bands
        for (int band = 0; band < numBands; ++band) {
            // Calculate band frequency range
            int startIdx = static_cast<int>(pow(band / (double)numBands, 2.0) * (N/4));
            int endIdx = static_cast<int>(pow((band + 1) / (double)numBands, 2.0) * (N/4));
            
            // In isolation mode, check if this band contains an active peak
            bool bandActive = !isolationMode;  // Always active if not in isolation mode
            if (isolationMode) {
                double bandStartFreq = binToFrequency(startIdx, wav->header.sampleRate, N);
                double bandEndFreq = binToFrequency(endIdx, wav->header.sampleRate, N);
                
                for (const auto& peak : activePeaks) {
                    if (peak.frequency >= bandStartFreq && peak.frequency <= bandEndFreq) {
                        bandActive = true;
                        break;
                    }
                }
            }

            // Skip inactive bands in isolation mode
            if (!bandActive) {
                for (int sub = 0; sub < subBands; ++sub) {
                    int idx = band * subBands + sub;
                    smoothedHeights[idx] *= 0.8;  // Gentle fadeout for inactive bands
                }
                continue;
            }

            // Calculate average magnitude for this band
            double bandEnergy = 0.0;
            int samplesCount = 0;
            
            // Exponential frequency mapping for better low-frequency resolution
            for (int i = startIdx; i < endIdx && i < magnitudesDB.size(); ++i) {
                bandEnergy += magnitudesDB[i];
                samplesCount++;
            }
            
            double avgDB = samplesCount > 0 ? bandEnergy / samplesCount : minDB;
            
            // Generate interpolated sub-bands
            for (int sub = 0; sub < subBands; ++sub) {
                int idx = band * subBands + sub;
                
                // Normalize and apply frequency-dependent scaling
                double normalizedHeight = (avgDB - minDB) / (maxDB - minDB);
                normalizedHeight = clamp(normalizedHeight, 0.0, 1.0);
                
                // Apply frequency-dependent scaling
                double freqScale = 1.0 - (0.3 * band / numBands);  // Boost lower frequencies
                normalizedHeight *= freqScale;
                
                // Apply nonlinear scaling for better visual response
                normalizedHeight = pow(normalizedHeight, 0.5);  // Adjusted response curve
                
                // Update target heights with interpolation
                targetHeights[idx] = normalizedHeight;
                
                // Apply asymmetric smoothing (different attack/release times)
                double smoothingFactor;
                if (targetHeights[idx] > smoothedHeights[idx]) {
                    smoothingFactor = attackTime;  // Fast attack
                } else {
                    smoothingFactor = releaseTime; // Slow release
                }
                
                smoothedHeights[idx] = smoothedHeights[idx] * (1.0 - smoothingFactor) + 
                                     targetHeights[idx] * smoothingFactor;
                
                // Calculate interpolated height
                int height = max(1, static_cast<int>(smoothedHeights[idx] * graphHeight * 0.95));
                
                // Enhanced color gradient
                double hue = 240.0 * (1.0 - smoothedHeights[idx]);  // Blue to Red
                double saturation = 0.8 + (0.2 * smoothedHeights[idx]);
                double value = 0.7 + (0.3 * smoothedHeights[idx]);
                
                // Convert HSV to RGB
                int r, g, b;
                HSVtoRGB(hue, saturation, value, r, g, b);
                
                // Draw main bar with anti-aliasing effect
                SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                SDL_Rect bar = {
                    idx * barWidth,
                    graphHeight - height,
                    max(3, barWidth - 1),  // Slightly narrower for gap
                    height
                };
                SDL_RenderFillRect(renderer, &bar);
                
                // Modify the color calculation to highlight active notes in isolation mode
                if (isolationMode && bandActive) {
                    int r, g, b;
                    HSVtoRGB(120.0, 0.9, 0.9, r, g, b);  // Green with high saturation and value
                    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                    SDL_RenderFillRect(renderer, &bar);  // Redraw the bar with new color
                }

                // Add bloom effect at the top
                SDL_SetRenderDrawColor(renderer, 
                    min(255, r + 40),
                    min(255, g + 40),
                    min(255, b + 40), 
                    180);
                SDL_Rect bloom = {
                    idx * barWidth,
                    graphHeight - height,
                    max(3, barWidth - 1),
                    min(8, height/4)
                };
                SDL_RenderFillRect(renderer, &bloom);
            }

            // Modify the color calculation to highlight active notes in isolation mode
            if (isolationMode && bandActive) {
                // Convert HSV to RGB for active notes
                int r, g, b;
                HSVtoRGB(120.0, 0.9, 0.9, r, g, b);  // Green with high saturation and value
                SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            }
        }
        
        // Define frequency points for labels (Hz)
        const vector<int> labelFreqs = {20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000};
        
        // Calculate the pixel position for each frequency
        for (int freq : labelFreqs) {
            // Convert frequency to log scale position
            double logFreq = log10(freq / minFreq) / log10(maxFreq / minFreq);
            int xPos = static_cast<int>(logFreq * (windowWidth - 20));  // 20px padding
            
            // Format the frequency label
            string label;
            if (freq >= 1000) {
                label = to_string(freq/1000) + "k";
            } else {
                label = to_string(freq);
            }
            
            // Render frequency value
            SDL_Color labelColor = {200, 200, 200, 255};  // Light grey color
            
            // Position the label centered below its frequency position
            SDL_Surface* textSurface = TTF_RenderText_Blended(font, label.c_str(), labelColor);
            if (textSurface) {
                int labelWidth = textSurface->w;
                int labelX = xPos - (labelWidth / 2);  // Center the label
                int labelY = windowHeight - 25;  // Position above bottom edge
                
                SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, textSurface);
                if (texture) {
                    SDL_Rect dstRect = {labelX, labelY, labelWidth, textSurface->h};
                    SDL_RenderCopy(renderer, texture, NULL, &dstRect);
                    SDL_DestroyTexture(texture);
                }
                SDL_FreeSurface(textSurface);
            }
            
            // Draw small tick mark above the label
            SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
            SDL_RenderDrawLine(renderer, 
                xPos, windowHeight - 30,  // Start of tick mark
                xPos, windowHeight - 35); // End of tick mark
        }

        // Add isolation mode indicator in top-right corner instead of top-left
        if (isolationMode) {
            string modeText = "Isolation Mode";
            // Calculate position to be in top-right corner with some padding
            int textX = windowWidth - 150;  // Adjust based on text width
            int textY = 10;  // Same vertical padding as before
            SDL_Color modeColor = {0, 255, 0, 255};
            renderText(modeText, textX, textY, modeColor);
        }
    }

    // Add this helper function for HSV to RGB conversion
    void HSVtoRGB(double h, double s, double v, int& r, int& g, int& b) {
        double c = v * s;
        double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
        double m = v - c;
        
        double r1, g1, b1;
        if (h < 60) {
            r1 = c; g1 = x; b1 = 0;
        } else if (h < 120) {
            r1 = x; g1 = c; b1 = 0;
        } else if (h < 180) {
            r1 = 0; g1 = c; b1 = x;
        } else if (h < 240) {
            r1 = 0; g1 = x; b1 = c;
        } else if (h < 300) {
            r1 = x; g1 = 0; b1 = c;
        } else {
            r1 = c; g1 = 0; b1 = x;
        }
        
        r = static_cast<int>((r1 + m) * 255);
        g = static_cast<int>((g1 + m) * 255);
        b = static_cast<int>((b1 + m) * 255);
    }

    void togglePause() {
        SDL_LockMutex(audioMutex);
        isPaused = !isPaused;
        SDL_UnlockMutex(audioMutex);
    }

public:
    AudioVisualizer(int width = 800, int height = 600)
        : windowWidth(width), windowHeight(height), framePointer(0) {
        initSDL();

        bands.clear();

        // Generate logarithmic bands
        for (int i = 0; i < numBands; i++) {
            double startFreq = minFreq * pow(freqMultiplier, i);
            double endFreq = startFreq * freqMultiplier;
            
            // Create 4 logarithmically-spaced subbands within each band
            vector<FrequencySubBand> subBands;
            double subMultiplier = pow(endFreq / startFreq, 1.0 / 4);
            
            for (int j = 0; j < 4; j++) {
                double subStart = startFreq * pow(subMultiplier, j);
                double subEnd = subStart * subMultiplier;
                subBands.push_back({
                    static_cast<int>(subStart),
                    static_cast<int>(subEnd),
                    0.0,
                    0.0
                });
            }

            // Enhanced color scheme for better visibility
            SDL_Color color;
            if (startFreq < 250) {  // Bass frequencies (red)
                color = {255, 0, 0, 255};
            } else if (startFreq < 2000) {  // Mid frequencies (green)
                color = {0, 255, 0, 255};
            } else {  // High frequencies (blue)
                color = {0, 0, 255, 255};
            }

            // Format frequency label
            string label;
            if (startFreq < 1000) {
                label = to_string(static_cast<int>(startFreq)) + "Hz";
            } else {
                label = to_string(static_cast<int>(startFreq / 1000)) + "kHz";
            }

            bands.push_back({
                static_cast<int>(startFreq),
                static_cast<int>(endFreq),
                0.0,
                0.0,
                color,
                label,
                subBands
            });
        }

        // Initialize active notes
        activeNotes.clear();
        for (const auto& note : NOTES) {
            activeNotes.push_back({
                note.name,
                note.frequency,
                0.0,
                false,
                0
            });
        }

        cout << "Initialized " << activeNotes.size() << " notes for tracking." << endl;

        audioMutex = SDL_CreateMutex();
        if (!audioMutex) {
            throw runtime_error("Failed to create audio mutex");
        }
    }

    ~AudioVisualizer() {
        if (audioMutex) {
            SDL_DestroyMutex(audioMutex);
        }
        if (audioDevice != 0) {
            SDL_CloseAudioDevice(audioDevice);
        }
        TTF_CloseFont(font);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
    }

    void visualize(const WavFile& wavFile, int fftWindowSize, int stepSize) {
        wav = &wavFile;
        initAudio(wavFile);

        vector<complex<double>> fftData(N);
        framePointer = 0;
        
        bool quit = false;
        SDL_Event e;

        // Timing variables for consistent frame rate
        const int targetFPS = 60;
        const int frameDelay = 1000 / targetFPS;
        Uint32 frameStart;
        int frameTime;

        while (!quit) {
            frameStart = SDL_GetTicks();

            while (SDL_PollEvent(&e) != 0) {
                if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) {
                    quit = true;
                }
                else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_SPACE) {
                    togglePause();
                    cout << (isPaused ? "Paused" : "Resumed") << endl;
                }
                else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_i) {
                    isolationMode = !isolationMode;
                }
            }

            if (!isPaused) {
                processFrame(wavFile, fftData);
            }
            
            SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
            SDL_RenderClear(renderer);
            
            renderSpectrum();
            renderNotes();
            
            // Add pause indicator
            if (isPaused) {
                SDL_Color pauseColor = {255, 255, 255, 255};
                renderText("PAUSED", windowWidth/2 - 50, windowHeight/2 - 20, pauseColor);
            }
            
            SDL_RenderPresent(renderer);
            
            // Update frame pointer only when not paused
            if (!isPaused) {
                framePointer += N/8;
                if (framePointer + N >= wavFile.samples.size()) {
                    framePointer = 0;
                }
            }

            // Cap frame rate
            frameTime = SDL_GetTicks() - frameStart;
            if (frameDelay > frameTime) {
                SDL_Delay(frameDelay - frameTime);
            }
        }
    }
};

// Helper function to ensure window size is a power of 2
int getNextPowerOf2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

string frequencyToNote(double frequency) {
    const vector<string> noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    double a4 = 440.0; // A4 reference frequency
    double semitones = 12 * log2(frequency / a4);
    int noteIndex = round(semitones) + 9; // A4 is at index 9
    
    int octave = 4 + (noteIndex / 12);
    noteIndex = ((noteIndex % 12) + 12) % 12;
    
    return noteNames[noteIndex] + to_string(octave);
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            cerr << "Usage: " << argv[0] << " <path to WAV file>" << endl;
            return 1;
        }

        // Test note identification
        cout << "Testing note identification:" << endl;
        vector<double> testFreqs = {440.0, 261.63, 329.63, 392.0, 523.25};
        for (double freq : testFreqs) {
            cout << "Frequency " << freq << "Hz -> Note: " << identifyNote(freq) << endl;
        }
        cout << "\n";

        // Load WAV file
        string filename = argv[1];
        WavFile wav = loadWav(filename);

        // Configure FFT parameters
        int windowSize = getNextPowerOf2(1024); // Ensure power of 2
        int stepSize = windowSize / 2; // 50% overlap

        // Print audio file information
        cout << "Audio Information:" << endl;
        cout << "Sample Rate: " << wav.header.sampleRate << " Hz" << endl;
        cout << "Channels: " << wav.header.numChannels << endl;
        cout << "Bit Depth: " << wav.header.bitsPerSample << " bits" << endl;
        cout << "Duration: " << wav.samples.size() / wav.header.sampleRate << " seconds" << endl;

        // Create and run visualizer
        AudioVisualizer visualizer(1024, 768); // Larger window for better visibility
        visualizer.visualize(wav, windowSize, stepSize);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
