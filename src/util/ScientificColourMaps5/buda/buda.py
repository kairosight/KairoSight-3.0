# 
#         buda
#                   www.fabiocrameri.ch/visualisation
from matplotlib.colors import LinearSegmentedColormap      
      
cm_data = [[0.70015, 0.0027445, 0.70061],      
           [0.70019, 0.010833, 0.69719],      
           [0.70023, 0.019196, 0.69378],      
           [0.70026, 0.027497, 0.69041],      
           [0.70028, 0.036129, 0.68707],      
           [0.7003, 0.044535, 0.68375],      
           [0.70032, 0.052201, 0.68047],      
           [0.70033, 0.059479, 0.67723],      
           [0.70034, 0.066138, 0.67402],      
           [0.70035, 0.0725, 0.67086],      
           [0.70035, 0.078557, 0.66775],      
           [0.70036, 0.084489, 0.66466],      
           [0.70037, 0.090118, 0.66165],      
           [0.70038, 0.095602, 0.65866],      
           [0.70039, 0.10092, 0.65574],      
           [0.7004, 0.10618, 0.65285],      
           [0.70042, 0.11127, 0.65003],      
           [0.70045, 0.11628, 0.64726],      
           [0.70048, 0.12114, 0.64453],      
           [0.70052, 0.12603, 0.64186],      
           [0.70057, 0.13079, 0.63925],      
           [0.70064, 0.13546, 0.63668],      
           [0.70073, 0.14008, 0.63417],      
           [0.70083, 0.14469, 0.6317],      
           [0.70095, 0.14919, 0.62928],      
           [0.7011, 0.15364, 0.62692],      
           [0.70128, 0.15808, 0.62461],      
           [0.70149, 0.16247, 0.62235],      
           [0.70173, 0.1668, 0.62015],      
           [0.70201, 0.17108, 0.61798],      
           [0.70234, 0.17534, 0.61587],      
           [0.70271, 0.17955, 0.6138],      
           [0.70312, 0.18374, 0.61179],      
           [0.70358, 0.18794, 0.60982],      
           [0.70409, 0.19203, 0.60791],      
           [0.70466, 0.19615, 0.60603],      
           [0.70527, 0.2002, 0.60421],      
           [0.70595, 0.2043, 0.60242],      
           [0.70667, 0.20831, 0.60069],      
           [0.70744, 0.21233, 0.59899],      
           [0.70826, 0.21633, 0.59734],      
           [0.70914, 0.2203, 0.59571],      
           [0.71005, 0.22422, 0.59413],      
           [0.71101, 0.22819, 0.59259],      
           [0.71201, 0.2321, 0.59107],      
           [0.71304, 0.236, 0.58959],      
           [0.71411, 0.23986, 0.58813],      
           [0.71521, 0.24372, 0.58671],      
           [0.71634, 0.24759, 0.58531],      
           [0.71749, 0.25142, 0.58392],      
           [0.71864, 0.25523, 0.58256],      
           [0.71983, 0.25903, 0.58122],      
           [0.72101, 0.2628, 0.5799],      
           [0.72221, 0.26656, 0.57858],      
           [0.72342, 0.27032, 0.57728],      
           [0.72464, 0.27403, 0.57601],      
           [0.72586, 0.27775, 0.57472],      
           [0.72708, 0.28143, 0.57347],      
           [0.72829, 0.28511, 0.57222],      
           [0.72952, 0.28879, 0.57097],      
           [0.73073, 0.29243, 0.56973],      
           [0.73195, 0.29606, 0.5685],      
           [0.73316, 0.29969, 0.56729],      
           [0.73437, 0.30328, 0.56607],      
           [0.73557, 0.30688, 0.56485],      
           [0.73677, 0.31047, 0.56366],      
           [0.73796, 0.31402, 0.56246],      
           [0.73916, 0.31757, 0.56126],      
           [0.74035, 0.32111, 0.56008],      
           [0.74153, 0.32463, 0.55888],      
           [0.74271, 0.32815, 0.55771],      
           [0.74389, 0.33165, 0.55653],      
           [0.74506, 0.33517, 0.55536],      
           [0.74622, 0.33863, 0.5542],      
           [0.74739, 0.34211, 0.55303],      
           [0.74855, 0.34556, 0.55187],      
           [0.7497, 0.34902, 0.55071],      
           [0.75087, 0.35247, 0.54956],      
           [0.75202, 0.35591, 0.54842],      
           [0.75316, 0.35932, 0.54726],      
           [0.75431, 0.36274, 0.54612],      
           [0.75545, 0.36616, 0.54498],      
           [0.75659, 0.36956, 0.54384],      
           [0.75773, 0.37295, 0.54271],      
           [0.75886, 0.37635, 0.54156],      
           [0.75999, 0.37972, 0.54044],      
           [0.76112, 0.3831, 0.53931],      
           [0.76225, 0.38646, 0.53819],      
           [0.76337, 0.38982, 0.53706],      
           [0.7645, 0.39318, 0.53593],      
           [0.76562, 0.39653, 0.53482],      
           [0.76673, 0.39988, 0.53369],      
           [0.76785, 0.40323, 0.53259],      
           [0.76897, 0.40656, 0.53147],      
           [0.77008, 0.4099, 0.53036],      
           [0.7712, 0.41322, 0.52925],      
           [0.7723, 0.41654, 0.52814],      
           [0.77341, 0.41986, 0.52704],      
           [0.77451, 0.42317, 0.52593],      
           [0.77562, 0.42649, 0.52482],      
           [0.77673, 0.4298, 0.52372],      
           [0.77783, 0.4331, 0.52262],      
           [0.77892, 0.43641, 0.52152],      
           [0.78002, 0.43971, 0.52043],      
           [0.78111, 0.44302, 0.51934],      
           [0.7822, 0.44631, 0.51826],      
           [0.78329, 0.44961, 0.51717],      
           [0.78437, 0.45291, 0.51608],      
           [0.78545, 0.4562, 0.51501],      
           [0.78653, 0.4595, 0.51394],      
           [0.78759, 0.46279, 0.51286],      
           [0.78866, 0.4661, 0.5118],      
           [0.78971, 0.46941, 0.51074],      
           [0.79076, 0.4727, 0.5097],      
           [0.79181, 0.47602, 0.50866],      
           [0.79285, 0.47933, 0.50762],      
           [0.79388, 0.48264, 0.50659],      
           [0.79489, 0.48595, 0.50557],      
           [0.7959, 0.48927, 0.50457],      
           [0.7969, 0.49259, 0.50356],      
           [0.79789, 0.49592, 0.50257],      
           [0.79888, 0.49926, 0.5016],      
           [0.79985, 0.5026, 0.50062],      
           [0.80081, 0.50594, 0.49965],      
           [0.80176, 0.5093, 0.49871],      
           [0.80271, 0.51265, 0.49778],      
           [0.80364, 0.516, 0.49683],      
           [0.80457, 0.51938, 0.49591],      
           [0.80548, 0.52274, 0.495],      
           [0.8064, 0.52612, 0.49409],      
           [0.8073, 0.5295, 0.49318],      
           [0.80819, 0.53287, 0.49229],      
           [0.80908, 0.53625, 0.49142],      
           [0.80996, 0.53963, 0.49053],      
           [0.81084, 0.54303, 0.48964],      
           [0.81172, 0.54642, 0.48878],      
           [0.81258, 0.54981, 0.48792],      
           [0.81345, 0.5532, 0.48705],      
           [0.81431, 0.5566, 0.48618],      
           [0.81518, 0.56, 0.48532],      
           [0.81604, 0.56341, 0.48446],      
           [0.8169, 0.56681, 0.4836],      
           [0.81775, 0.5702, 0.48275],      
           [0.81861, 0.57361, 0.4819],      
           [0.81946, 0.57702, 0.48103],      
           [0.82032, 0.58043, 0.48019],      
           [0.82118, 0.58383, 0.47934],      
           [0.82204, 0.58724, 0.47848],      
           [0.82289, 0.59065, 0.47763],      
           [0.82374, 0.59407, 0.47677],      
           [0.8246, 0.59749, 0.47592],      
           [0.82545, 0.6009, 0.47508],      
           [0.82631, 0.60433, 0.47421],      
           [0.82717, 0.60775, 0.47335],      
           [0.82802, 0.61117, 0.4725],      
           [0.82888, 0.61459, 0.47165],      
           [0.82974, 0.61802, 0.4708],      
           [0.8306, 0.62145, 0.46994],      
           [0.83146, 0.62489, 0.46909],      
           [0.83232, 0.62832, 0.46822],      
           [0.83317, 0.63176, 0.46737],      
           [0.83404, 0.6352, 0.46651],      
           [0.8349, 0.63864, 0.46564],      
           [0.83576, 0.64209, 0.46479],      
           [0.83662, 0.64553, 0.46393],      
           [0.83749, 0.64899, 0.46306],      
           [0.83835, 0.65244, 0.4622],      
           [0.83921, 0.6559, 0.46134],      
           [0.84008, 0.65936, 0.46047],      
           [0.84095, 0.66283, 0.45961],      
           [0.84182, 0.66629, 0.45875],      
           [0.84268, 0.66976, 0.45788],      
           [0.84355, 0.67324, 0.45701],      
           [0.84442, 0.67671, 0.45615],      
           [0.84529, 0.68019, 0.45528],      
           [0.84616, 0.68368, 0.45442],      
           [0.84703, 0.68717, 0.45354],      
           [0.84791, 0.69065, 0.45268],      
           [0.84878, 0.69415, 0.4518],      
           [0.84966, 0.69765, 0.45092],      
           [0.85053, 0.70116, 0.45005],      
           [0.85141, 0.70467, 0.44918],      
           [0.8523, 0.70818, 0.4483],      
           [0.85317, 0.7117, 0.44741],      
           [0.85405, 0.71522, 0.44654],      
           [0.85494, 0.71874, 0.44566],      
           [0.85582, 0.72227, 0.44479],      
           [0.8567, 0.72581, 0.44391],      
           [0.85758, 0.72935, 0.44302],      
           [0.85847, 0.73289, 0.44214],      
           [0.85936, 0.73644, 0.44125],      
           [0.86025, 0.74, 0.44036],      
           [0.86114, 0.74355, 0.43947],      
           [0.86203, 0.74711, 0.43858],      
           [0.86292, 0.75068, 0.43771],      
           [0.86382, 0.75426, 0.43681],      
           [0.86471, 0.75783, 0.43592],      
           [0.8656, 0.76141, 0.43503],      
           [0.8665, 0.76501, 0.43412],      
           [0.8674, 0.76859, 0.43323],      
           [0.86831, 0.77219, 0.43234],      
           [0.86921, 0.7758, 0.43144],      
           [0.8701, 0.77941, 0.43054],      
           [0.87101, 0.78302, 0.42964],      
           [0.87192, 0.78665, 0.42873],      
           [0.87283, 0.79027, 0.42783],      
           [0.87374, 0.79391, 0.42693],      
           [0.87466, 0.79754, 0.42602],      
           [0.87558, 0.80119, 0.42512],      
           [0.8765, 0.80485, 0.4242],      
           [0.87744, 0.80851, 0.42329],      
           [0.87838, 0.81218, 0.42239],      
           [0.87933, 0.81586, 0.42148],      
           [0.8803, 0.81954, 0.42058],      
           [0.88128, 0.82324, 0.41968],      
           [0.88228, 0.82696, 0.41878],      
           [0.8833, 0.83068, 0.41788],      
           [0.88436, 0.83442, 0.41699],      
           [0.88544, 0.83817, 0.4161],      
           [0.88657, 0.84195, 0.41522],      
           [0.88774, 0.84574, 0.41435],      
           [0.88898, 0.84956, 0.4135],      
           [0.89027, 0.8534, 0.41266],      
           [0.89165, 0.85727, 0.41182],      
           [0.8931, 0.86118, 0.41101],      
           [0.89465, 0.86511, 0.41022],      
           [0.8963, 0.86908, 0.40944],      
           [0.89805, 0.87309, 0.40869],      
           [0.89994, 0.87713, 0.40798],      
           [0.90195, 0.88122, 0.40729],      
           [0.9041, 0.88535, 0.40663],      
           [0.9064, 0.88953, 0.40601],      
           [0.90885, 0.89375, 0.40542],      
           [0.91146, 0.89801, 0.40486],      
           [0.91422, 0.90232, 0.40435],      
           [0.91714, 0.90667, 0.40388],      
           [0.92022, 0.91107, 0.40344],      
           [0.92346, 0.9155, 0.40303],      
           [0.92684, 0.91998, 0.40266],      
           [0.93038, 0.92449, 0.40233],      
           [0.93405, 0.92904, 0.40204],      
           [0.93787, 0.93363, 0.40178],      
           [0.9418, 0.93824, 0.40155],      
           [0.94585, 0.94289, 0.40136],      
           [0.95, 0.94756, 0.40118],      
           [0.95426, 0.95224, 0.40103],      
           [0.9586, 0.95696, 0.4009],      
           [0.96301, 0.96169, 0.40079],      
           [0.9675, 0.96644, 0.40069],      
           [0.97205, 0.97121, 0.4006],      
           [0.97664, 0.97598, 0.40052],      
           [0.98129, 0.98078, 0.40045],      
           [0.98597, 0.98558, 0.40038],      
           [0.99068, 0.9904, 0.40032],      
           [0.99542, 0.99522, 0.40026],      
           [1, 1, 0.4002]]      
      
buda_map = LinearSegmentedColormap.from_list('buda', cm_data)      
# For use of "viscm view"      
test_cm = buda_map      
      
if __name__ == "__main__":      
    import matplotlib.pyplot as plt      
    import numpy as np      
      
    try:      
        from viscm import viscm      
        viscm(buda_map)      
    except ImportError:      
        print("viscm not found, falling back on simple display")      
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',      
                   cmap=buda_map)      
    plt.show()      
