import audio_functions as auf
import plot
import torch
import torchaudio

def Simple_IR_Gen(t60, sound_rays_data, ts=50, DEVICE="cpu", fs=44100):
    """
    Generates synthetic room impulse response signal.
    Input:
        - t60: float type object. Expected reverberation time by Sabine's equation.
        - sound_rays_data: list type object. Contains information about direct sound and early reflections:
            - arrival_time [ms]: float type object.
            - amplitude: float type object
        - ts: float type object. Time [ms] between early response and late response
        - device: processing type for pytorch
        - fs: Sampling frequency
    """

    # Early Response
    he = torch.zeros(t60*fs + 1, device=DEVICE)
    time_scale = torch.linspace(len(he), step=1/fs)

    for t, A in sound_rays_data:
        index = torch.argmin(torch.abs(time_scale - t/1000))
        he[index] = A

    # Late Response
    noise = torch.normal(mean=0.0, std=torch.ones_like(he))
    tao = t60/6.90
    envelope = torch.exp(-time_scale/tao)
    hl = envelope*noise

    hl = hl/(torch.max(hl))
    separation_index = torch.argmin(torch.abs(time_scale - ts/1000))

    IR = torch.cat(he[:ts], hl[ts:])

    return IR



    