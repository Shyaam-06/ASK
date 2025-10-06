# ASK AND FSK
# Aim
Write a simple Python program for the modulation and demodulation of ASK and FSK.
# Tools required
Python IDE with Numpy and Scipy
# Program
ASK 
```python
import numpy as np
 import matplotlib.pyplot as plt
 from scipy.signal import butter, lfilter
 # Butterworth low-pass filter for demodulation
 def butter_lowpass_filter(data, cutoff, fs, order=5):
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return lfilter(b, a, data)
 # Parameters
 fs = 1000 
f_carrier = 50 
bit_rate = 10 

 T = 1 
t = np.linspace(0, T, int(fs * T), endpoint=False)
 # Message signal (binary data)
 bits = np.random.randint(0, 2, bit_rate)
 bit_duration = fs // bit_rate
 message_signal = np.repeat(bits, bit_duration)
 # Carrier signal
 carrier = np.sin(2 * np.pi * f_carrier * t)
 # ASK Modulation
 ask_signal = message_signal * carrier
 # ASK Demodulation
 demodulated = ask_signal * carrier 
filtered_signal = butter_lowpass_filter(demodulated, f_carrier, fs)
 decoded_bits = (filtered_signal[::bit_duration] > 0.25).astype(int)
 # Plotting
 plt.figure(figsize=(12, 8))
 plt.subplot(4, 1, 1)
 plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
 plt.title('Message Signal')
 plt.grid(True)
 plt.subplot(4, 1, 2)
 plt.plot(t, carrier, label='Carrier Signal', color='g')
 plt.title('Carrier Signal')
 plt.grid(True)
 plt.subplot(4, 1, 3)
 plt.plot(t, ask_signal, label='ASK Modulated Signal', color='r')
 plt.title('ASK Modulated Signal')
 plt.grid(True)
 plt.subplot(4, 1, 4)
 plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='r', 
marker='x')
 plt.title('Decoded Bits')
 plt.tight_layout()
 plt.show()
```
FSK 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- Filter Function Definition (from image 2) ---

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the input data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Butterworth filter design
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    return lfilter(b, a, data)

# --- FSK Simulation Parameters (from image 3) ---

fs = 1000  # Sampling frequency
f1 = 30    # Carrier frequency for bit 0
f2 = 70    # Carrier frequency for bit 1
T = 1      # Total simulation time in seconds

t = np.linspace(0, T, int(fs * T), endpoint=False)

# --- Message Signal Generation (from image 3) ---

bit_rate = 10
# Generate 8 random bits (0s and 1s)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
# Create the message signal by repeating each bit for its duration
message_signal = np.repeat(bits, bit_duration)

# --- Carrier Signals (from image 3) ---

# Carrier for bit 0
carrier_f1 = np.sin(2 * np.pi * f1 * t)
# Carrier for bit 1
carrier_f2 = np.sin(2 * np.pi * f2 * t)

# --- FSK Modulation (from image 3) ---

fsk_signal = np.zeros(t.shape)

for i, bit in enumerate(bits):
    start = i * bit_duration
    end = start + bit_duration
    
    # Select frequency based on the bit
    freq = f2 if bit else f1
    
    # Generate the sine wave for the current bit duration
    # Note: The original code uses 't[start:end]' in the sin function, 
    # which is often incorrect for continuous phase. A cleaner implementation
    # uses a local time vector or correctly offsets 't', but following the 
    # provided image code precisely:
    fsk_signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

# --- Reference Signals for Demodulation (from image 3) ---

# Reference sine waves for correlation at f1 and f2
ref_f1 = np.sin(2 * np.pi * f1 * t)
ref_f2 = np.sin(2 * np.pi * f2 * t)

# --- FSK Demodulation (Coherent Correlation Method) (from image 3) ---

# Correlate FSK signal with reference signals and low-pass filter the result
# The correlation is achieved by multiplication followed by low-pass filtering.
corr_f1 = butter_lowpass_filter(fsk_signal * ref_f1, f1, fs) # Cutoff at f1
corr_f2 = butter_lowpass_filter(fsk_signal * ref_f2, f2, fs) # Cutoff at f2

decoded_bits = []
for i in range(bit_rate):
    start = i * bit_duration
    end = start + bit_duration
    
    # Calculate energy (sum of squares) of the correlated signals over one bit duration
    energy_f1 = np.sum(corr_f1[start:end] ** 2)
    energy_f2 = np.sum(corr_f2[start:end] ** 2)
    
    # Decision: if energy at f2 > energy at f1, it's bit '1', otherwise '0'
    decoded_bits.append(1 if energy_f2 > energy_f1 else 0)

# Create the demodulated signal for plotting
decoded_bits = np.array(decoded_bits)
demodulated_signal = np.repeat(decoded_bits, bit_duration)

# --- Plotting (from image 3 and image 1) ---

plt.figure(figsize=(12, 12)) # Set figure size

# Plot 1: Message Signal (from image 3)
plt.subplot(6, 1, 1)
plt.plot(t, message_signal, color='b')
plt.title('Message Signal')
plt.grid(True)

# Plot 2: Carrier Signal for bit 0 (f1) (from image 3)
plt.subplot(6, 1, 2)
plt.plot(t, carrier_f1, color='g')
plt.title('Carrier Signal for bit = 0 (f1)')
plt.grid(True)

# Plot 3: Carrier Signal for bit 1 (f2) (from image 1)
plt.subplot(6, 1, 3)
plt.plot(t, carrier_f2, color='r')
plt.title('Carrier Signal for bit = 1 (f2)')
plt.grid(True)

# Plot 4: FSK Modulated Signal (from image 1)
plt.subplot(6, 1, 4)
plt.plot(t, fsk_signal, color='m')
plt.title('FSK Modulated Signal')
plt.grid(True)

# Plot 5: Final Demodulated Signal (from image 1)
# Note: This is the regenerated message signal based on the demodulation result.
plt.subplot(6, 1, 5)
plt.plot(t, demodulated_signal, color='k')
plt.title('Final Demodulated Signal')
plt.grid(True)

# Adjust layout to prevent subplots from overlapping and display the figure
plt.tight_layout()
plt.show()
```
# Output Waveform
ASK
<img width="1190" height="790" alt="image" src="https://github.com/user-attachments/assets/7f672c77-5448-4a94-9b4a-26ac936defdd" />
FSK
<img width="1201" height="1012" alt="image" src="https://github.com/user-attachments/assets/a02b00be-946d-48e1-8c1d-f7ce6546f05f" />


# Results

Python program for the modulation and demodulation of ASK and FSK has been executed. 

# Hardware experiment output waveform.
