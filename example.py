import audio_functions as auf
import IR_generation as IRg

"""Example: """

sound_rays_array = [
  [6.62, 15.7e-3],   #SD
  [7.10, 7.78e-3],   #R11
  [7.42, 7.12e-3],   #R12
  [15.14, 1.711e-3],  #R13
  [23.76, 6.95e-4],   #R14
  [7.8, 7.35e-3],   #R21
  [23.886, 7.84e-4],   #R22
  [27.37, 5.97e-4],   #R23
  [23.76, 7.92e-4],   #R24
  [15.904, 8.84e-4],   #R25
  [24.58, 3.7e-4],   #R26
  [21.816, 4.69e-4],   #R27
  [36.3, 1.69e-4],   #R28
  [37.16, 1.61e-4],   #R29
  [24.94, 4.18e-4],   #R31
  [28.065, 3.23e-4],   #R32
  [27.803, 3.29e-4],   #R33
  [16.228, 9.68e-4],   #R34
  [22.05, 5.24e-4],   #R35
  [38.797, 8.46e-5],   #R36
  [32.367, 1.21e-4],   #R37
  [53.64, 4.38e-5],   #R38
  [31.55, 1.28e-4],   #R39
  [28.50, 1.78e-4],   #R41
  [32,52, 6.37e-5],   #R42
  [39,1, 4.75e-5],   #R43
  [41.85, 4.14e-5],   #R44
  [39.59, 4.63e-5],   #R45
  [55.62, 2.34e-5],   #R46
  [32.17, 7.02e-5],   #R47
]

t60_sabine = 1.6

IR, he, hl = IRg.Simple_IR_Gen(t60_sabine, sound_rays_array)

auf.save_audio("synthetic_IR_1", IR)
auf.save_audio("synthetic_he_1", he)
auf.save_audio("synthetic_hl_1", hl)