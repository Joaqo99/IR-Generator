import soundfile as sf
from IPython.display import Audio
import numpy as np
from scipy import signal
import torchaudio
import torch
from torchaudio import transforms
import numpy as np

def conv(in_signal, ir):
    """Performs convolution"""
    return signal.fftconvolve(in_signal, ir, mode='same')

def moving_media_filter(in_signal, N):
    """Performs media moving filter"""
    ir = np.ones(N) * 1 / N
    return conv(in_signal, ir)

def synth_impulse_response(fs, reverb_time, noise_florr_level, A=1.0):
    """
    Generates a synthetic impulse response
    Input:
        - fs: int type object. Sample rate
        - reverb_time: float type object. Reverb time.
        - noise_florr_level: int type object. Noise floor presion level.
        - A: float type object. Exponential amplitude. Optional, 1.0 by default.
    Output:
        - t: array type object. Time vector
        - impulse_response: array type object. Impulse response vector
    """

    #error handling
    if type(fs) != int:
        raise ValueError("fs must be an integer")
    if type(reverb_time) != float:
        raise ValueError("reverb_time must be a float")
    if type(noise_florr_level) != int:
        raise ValueError("noise_floor_level must be a int")
    if type(A) != float:
        raise ValueError("A must be a float")

    #cómo genero n? --> n, t, lo q sea, es arbitrario. Tiene que ser mayor al tiempo de reverberación.
    dur = reverb_time + 0.25
    signal_length = int(dur*fs)
    t = np.linspace(0, dur, signal_length, endpoint=True)

    #generate noise
    noise = np.random.normal(0, 1, signal_length)

    #envelop
    tao = reverb_time/6.90
    envolvente = np.exp(-t/tao)

    #impulse response generator
    impulse_response = A*envolvente*noise + (10**(noise_florr_level/20))*noise
    impulse_response = impulse_response/ np.max(np.abs(impulse_response))

    return t, impulse_response

def load_audio(file_name, output_format="numpy"):
    """
    Loads a mono or stereo audio file in audios folder.
    Input:
        - file_name: str type object. The file must be an audio file.
        - output_format: str type object. The desired vector output format ('numpy' or 'torch'). Defaults to 'numpy'.
    Output:
        - audio: array type object.
        - fs: sample frequency
        - prints if audio is mono or stereo.
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    audio, fs = torchaudio.load(f"./audios/{file_name}")
    audio = audio.squeeze()

    # Converts the resampled signal to the desired output format
    if output_format == 'numpy':
        audio = audio.numpy().astype(np.float32)
    elif output_format == 'torch':
        audio = audio.type(torch.float32)

    return audio, fs

def play_audio(audio, fs):
    """
    Plays a mono audio
    Inputs:
        - audio: array type object. Audio to play. Must be mono.
        - fs: int type object. Sample rate
    """
    #error handling
    if type(fs) != int:
        raise ValueError("fs must be int")

    return Audio(audio, rate=fs)

def to_mono(audio):
    """
    Converts a stereo audio vector to mono.
    Insert:
        - audio: array type object of 2 rows. Audio to convert.
    Output:
        - audio_mono: audio converted
    """
    #error handling



    if  type(audio) != np.ndarray and type(audio) != torch.Tensor:
        raise ValueError("audio must be a vector")
    if len(audio.shape) == 1:
        raise Exception("Audio is already mono")
    elif audio.shape[0] != 2 and audio.shape[1] != 2: 
        raise Exception("Non valid vector")
    
    #features
    audio_mono = (audio[:,0]/2)+(audio[:,1]/2)
    return audio_mono

def reverb(ir, audio, ir_fs, audio_fs):
    """
    Returns an auralization of an audio and a given impulse response
    Input:
        - ir: array type object. Impulse response
        - audio: array type object. Must be mono audio.
        - ir_fs: int type object. Impulse response sample rate.
        - audio_fs: int type object. Audio sample rate.
    Output:
        - audio_auralized: array type object.
    """

    #error handling
    if type(ir) != np.ndarray or type(audio) != np.ndarray:
        raise ValueError("both audio and ir must be a ndarray")
    if type(ir_fs) != int or type(audio_fs) != int:
        raise ValueError("fs must be int")
    
    assert ir_fs == audio_fs, "Both Impulse Response and Audio sample rates must be the same"
    
    if len(audio.shape) != 1:
        raise Exception("Audio must be mono")
    
    #features
    audio_auralized = signal.fftconvolve(ir, audio)
    return audio_auralized

def get_audio_time_array(audio, fs):
    """
    Returns audio time array
    Input:
        - audio: array type object.
        - fs: Int type object. Sample rate.
    Output:
        - duration: int type object. Audio duration
        - time_array: array type object.
    """
    #error handling
    #if  type(audio) != np.ndarray or type():
    #    raise ValueError("audio must be a ndarray")
    if type(fs) != int:
        raise ValueError("fs must be int")
    
    #features
    duration = len(audio) // fs
    time_array = np.linspace(0, duration, len(audio))

    return duration, time_array

def to_dB(audio):
    """
    Returns an audio amplitude array in dB scale
    Input:
        - audio: array type object.
    Output:
        - audio_db: array type object.
    """
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    
    audio_db = 10*np.log10(audio**2)
    return audio_db

def generate_time_vector(dur, fs):
    """
    Generates a time vector:
    Inputs:
        - dur: float type object. Vector time duration
        - fs: int type object. Sample frequency.
    Outputs:
        - t: array type object. Time vector
    """

    t = np.linspace(0, dur, int(dur*fs))
    return t

def generate_cos(t, f=100, phase=0):
    """
    Generates a sinusoidal signal
    Inputs:
        - t: array type object. Input signal.
        - f: float type object. Frequency of signal in Hertz [Hz]. 100 Hz by default
        - phase: float type object. 0 by default
    Output:
        - sin_signal: array type object. Signal array
    """
    sin_arg = 2*np.pi*f*t
    sin_signal = np.cos(sin_arg + phase)
    return sin_signal

def resample_signal_fs(in_signal, original_sr, target_sr, output_format='numpy'):
    """
    Resamples a signal to a target sampling rate using torchaudio.

    Parameters:
        signal (torch.Tensor or np.ndarray): The input signal.
        original_sr (int): The original sampling rate of the input signal.
        target_sr (int): The target sampling rate for resampling.
        output_format (str, optional): The desired output format ('numpy' or 'torch').Defaults to 'numpy'.
        
    Returns:
        resampled_signal: The resampled signal in the specified format.
    """
    # Convert the input signal to a torch tensor if it's a NumPy array
    if not isinstance(original_sr, int) or not isinstance(target_sr, int):
        raise TypeError("Los parámetros original_sr y target_sr deben ser enteros.")

    if isinstance(in_signal, np.ndarray):
        in_signal = torch.from_numpy(in_signal)

    # Resample the signal
    if original_sr == target_sr:
        resampled_signal = in_signal
        print("Las frecuencias de sampleo son iguales, no es necesario resamplear")
    else:
        resampled_signal = torchaudio.transforms.Resample(original_sr, target_sr)(in_signal)
        print(f"Señal resampleada de {original_sr} Hz a {target_sr} Hz")

    # Convert the resampled signal to the desired output format
    if output_format == 'numpy':
        resampled_signal = resampled_signal.numpy().astype(np.float32)
    elif output_format == 'torch':
        resampled_signal = resampled_signal.type(torch.float32)

    return resampled_signal

def resample_signal_length(in_signal, target_len, output_format="numpy"):
    """
    Resamples a signal to a target length.

    Parameters:
        signal (torch.Tensor or np.ndarray): The input signal.
        target_len (int): The target signal length.
        output_format (str, optional): The desired output format ('numpy' or 'torch').Defaults to 'numpy'.
        
    Returns:
        resampled_signal: The resampled signal in the specified format.
    """

        # Convert the input signal to a torch tensor if it's a NumPy array
    if not isinstance(target_len, int):
        raise TypeError("target_length debe ser entero.")

    if isinstance(in_signal, torch.Tensor):
        in_signal = in_signal.numpy().astype(np.float32)

    original_len = len(in_signal)

        # Resample the signal
    if original_len == target_len:
        resampled_signal = in_signal
        print("Las longitudes son iguales, no es necesario resamplear")
    else:
        resampled_signal = signal.resample(in_signal, target_len)

    # Convert the resampled signal to the desired output format
    if output_format == 'numpy':
        pass
    elif output_format == 'torch':
        resampled_signal = torch.from_numpy(resampled_signal)
        resampled_signal = resampled_signal.type(torch.float32)

    return resampled_signal

def get_fft(in_signal, fs, normalize=True, output="mag-phase"):
    """
    Performs a fast fourier transform over the input signal. As we're working with real signals, we perform the rfft.
    Input:
        - in_signal: array or Torch tensor type object. input signal.
        - fs: int type object. Sample frequency
        - normalize: bool type object. If true, returns the normalized magnitude of the input signal. If output is "complex" it wont work
        - output: str type object. Output format, can be:
            - "mag-phase" for the magnitude and phase of the rfft. Default.
            - "complex" for the raw rfft.

    If Output = mag_phase:
        - in_freqs: array type object. Real Frequencies domain vector.
        - fft_mag: array type object. Real Frequencies amplitude vector.
        - fft_phase: array type object. RealFrequencies phase vector.
    If Output = complex:
        - in_freqs: array type object. Real Frequencies domain vector.
        - fft: array type object. Real Frequencies raw fft vector.
    """

    rfft = np.fft.rfft(in_signal)
    in_freqs = np.linspace(0, fs//2, len(rfft))

    #import pdb;pdb.set_trace()

    if output == "complex":
        return in_freqs, rfft
    elif output == "mag-phase":
        rfft_mag = abs(rfft)/len(rfft)
        if normalize: rfft_mag = rfft_mag / np.max(abs(rfft_mag))
        rfft_phase = np.angle(rfft)
        return in_freqs, rfft_mag, rfft_phase
    else:
        raise ValueError('No valid output format - Must be "mag-phase" or "complex"')

def get_ifft(in_rfft, in_phases=False, input="mag-phase"):
    """
    Performs an inverse fast Fourier transform of a real signal
    Input:
        - in_rfft_mag: array type object. It must contain only the positive frequencies of the spectrum of the signal.
        - in_phases: array type object. It must contain only the positive frequencies of the spectrum of the signal. If false, it assumes the phases of all components are 0º.
        - input: str type object. Input format, can be "mag-phase" or "complex", "mag_phase" by default. If "complex", there must not be in_phases kwarg.
    Output:
        - temp_signal: array type object. Transformed signal.
    """
    if input == "mag-phase":
        if type(in_phases) == bool and in_phases == False:
            in_phases = np.zeros(len(in_rfft))
    
        in_rfft = in_rfft * np.exp(1j * in_phases)
    elif input == "complex":
        if in_phases:
            raise Exception('If "complex" input there must not be a phase array input.')
    else:
        raise ValueError('Input format must be "mag_phase" or "complex"')
    
    temp_signal = np.fft.irfft(in_rfft)
    return temp_signal

def save_audio(file_name, audio, fs=44100):
    """
    Save an audio signal to a file in WAV format.

    Parameters:
        - file_name (str): Name of the output WAV file.
        - audio (ndarray): Audio signal to save.
        - fs (int, optional): Sampling rate. Default is 48000.

    Returns:
        None
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    sf.write(f"audios/{file_name}", audio, fs)
    print(f"Audio saved in audios/{file_name}")
    return 