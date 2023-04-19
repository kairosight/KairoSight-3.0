# 
#         roma
#                   www.fabiocrameri.ch/visualisation
from matplotlib.colors import LinearSegmentedColormap      
      
cm_data = [[0.49684, 0.099626, 0],      
           [0.50141, 0.11159, 0.0038271],      
           [0.50595, 0.12281, 0.0075362],      
           [0.51049, 0.13362, 0.011176],      
           [0.51502, 0.14397, 0.014691],      
           [0.51953, 0.15397, 0.017936],      
           [0.52403, 0.16373, 0.02102],      
           [0.52851, 0.17319, 0.023941],      
           [0.53298, 0.18247, 0.026768],      
           [0.53742, 0.19159, 0.02974],      
           [0.54181, 0.20053, 0.032886],      
           [0.5462, 0.20938, 0.036371],      
           [0.55055, 0.2181, 0.03985],      
           [0.55486, 0.22671, 0.043275],      
           [0.55915, 0.23522, 0.046908],      
           [0.56342, 0.24362, 0.050381],      
           [0.56764, 0.25199, 0.053844],      
           [0.57184, 0.26026, 0.057332],      
           [0.57601, 0.26848, 0.06081],      
           [0.58015, 0.27665, 0.064287],      
           [0.58425, 0.28474, 0.067738],      
           [0.58833, 0.29281, 0.071178],      
           [0.59239, 0.30083, 0.074547],      
           [0.59641, 0.30883, 0.077891],      
           [0.60041, 0.31677, 0.081344],      
           [0.60439, 0.32468, 0.084709],      
           [0.60834, 0.33258, 0.088059],      
           [0.61225, 0.34043, 0.091425],      
           [0.61616, 0.34827, 0.094762],      
           [0.62004, 0.35608, 0.098071],      
           [0.6239, 0.36387, 0.10135],      
           [0.62774, 0.37164, 0.10462],      
           [0.63157, 0.37941, 0.10794],      
           [0.63537, 0.38716, 0.11129],      
           [0.63916, 0.3949, 0.11452],      
           [0.64294, 0.40263, 0.11786],      
           [0.64671, 0.41037, 0.12112],      
           [0.65046, 0.41807, 0.12443],      
           [0.65421, 0.42579, 0.12776],      
           [0.65795, 0.43351, 0.13108],      
           [0.66169, 0.44123, 0.13442],      
           [0.66541, 0.44896, 0.13776],      
           [0.66914, 0.45668, 0.14112],      
           [0.67287, 0.46443, 0.1445],      
           [0.6766, 0.47218, 0.1479],      
           [0.68033, 0.47994, 0.15134],      
           [0.68407, 0.48772, 0.15484],      
           [0.68783, 0.49551, 0.1584],      
           [0.69159, 0.50332, 0.162],      
           [0.69537, 0.51117, 0.1656],      
           [0.69916, 0.51905, 0.16938],      
           [0.70298, 0.52695, 0.17315],      
           [0.70681, 0.53488, 0.17706],      
           [0.71067, 0.54285, 0.18103],      
           [0.71456, 0.55087, 0.18517],      
           [0.71848, 0.55892, 0.18939],      
           [0.72244, 0.56704, 0.19377],      
           [0.72644, 0.57519, 0.19826],      
           [0.73049, 0.5834, 0.20295],      
           [0.73457, 0.59168, 0.20781],      
           [0.7387, 0.60002, 0.21285],      
           [0.74289, 0.60842, 0.21813],      
           [0.74712, 0.61688, 0.22361],      
           [0.75142, 0.6254, 0.22933],      
           [0.75576, 0.634, 0.23533],      
           [0.76016, 0.64265, 0.24156],      
           [0.76463, 0.65137, 0.24809],      
           [0.76914, 0.66014, 0.2549],      
           [0.77371, 0.66897, 0.262],      
           [0.77833, 0.67784, 0.26943],      
           [0.783, 0.68674, 0.27716],      
           [0.78771, 0.69568, 0.2852],      
           [0.79246, 0.70462, 0.29358],      
           [0.79722, 0.71357, 0.30227],      
           [0.80201, 0.72249, 0.31128],      
           [0.80681, 0.73138, 0.32059],      
           [0.81159, 0.74021, 0.33017],      
           [0.81635, 0.74896, 0.34004],      
           [0.82108, 0.75761, 0.35015],      
           [0.82576, 0.76614, 0.36047],      
           [0.83037, 0.77452, 0.37103],      
           [0.8349, 0.78274, 0.38176],      
           [0.83934, 0.79077, 0.39264],      
           [0.84366, 0.7986, 0.40365],      
           [0.84785, 0.80619, 0.41475],      
           [0.8519, 0.81354, 0.42591],      
           [0.8558, 0.82064, 0.43711],      
           [0.85953, 0.82748, 0.44831],      
           [0.86308, 0.83404, 0.4595],      
           [0.86643, 0.84031, 0.47065],      
           [0.86958, 0.84629, 0.48173],      
           [0.87253, 0.85199, 0.49272],      
           [0.87526, 0.8574, 0.50362],      
           [0.87777, 0.86254, 0.51441],      
           [0.88004, 0.86739, 0.52506],      
           [0.88209, 0.87197, 0.53557],      
           [0.8839, 0.87629, 0.54595],      
           [0.88546, 0.88035, 0.55615],      
           [0.88677, 0.88417, 0.56622],      
           [0.88783, 0.88775, 0.57613],      
           [0.88864, 0.89111, 0.58587],      
           [0.88918, 0.89426, 0.59544],      
           [0.88946, 0.8972, 0.60485],      
           [0.88947, 0.89994, 0.61409],      
           [0.88921, 0.9025, 0.62319],      
           [0.88867, 0.90488, 0.6321],      
           [0.88785, 0.90709, 0.64085],      
           [0.88674, 0.90914, 0.64945],      
           [0.88534, 0.91104, 0.65787],      
           [0.88364, 0.91279, 0.66612],      
           [0.88165, 0.9144, 0.67421],      
           [0.87934, 0.91587, 0.68212],      
           [0.87673, 0.91722, 0.68988],      
           [0.87381, 0.91842, 0.69745],      
           [0.87058, 0.9195, 0.70485],      
           [0.86703, 0.92046, 0.71207],      
           [0.86316, 0.92129, 0.71912],      
           [0.85897, 0.92201, 0.72598],      
           [0.85447, 0.9226, 0.73266],      
           [0.84965, 0.92307, 0.73915],      
           [0.84452, 0.92342, 0.74544],      
           [0.83906, 0.92365, 0.75155],      
           [0.8333, 0.92375, 0.75746],      
           [0.82723, 0.92373, 0.76318],      
           [0.82086, 0.92358, 0.7687],      
           [0.81418, 0.9233, 0.77403],      
           [0.80722, 0.92289, 0.77916],      
           [0.79997, 0.92234, 0.7841],      
           [0.79243, 0.92166, 0.78883],      
           [0.78462, 0.92082, 0.79337],      
           [0.77654, 0.91986, 0.79771],      
           [0.7682, 0.91873, 0.80185],      
           [0.7596, 0.91747, 0.80581],      
           [0.75077, 0.91603, 0.80957],      
           [0.74169, 0.91444, 0.81313],      
           [0.7324, 0.91268, 0.81651],      
           [0.72287, 0.91075, 0.8197],      
           [0.71314, 0.90865, 0.8227],      
           [0.70322, 0.90636, 0.82551],      
           [0.69311, 0.90389, 0.82814],      
           [0.68283, 0.90124, 0.83059],      
           [0.67239, 0.89839, 0.83284],      
           [0.6618, 0.89535, 0.83492],      
           [0.65107, 0.89211, 0.83682],      
           [0.64024, 0.88868, 0.83853],      
           [0.6293, 0.88504, 0.84006],      
           [0.61828, 0.8812, 0.84141],      
           [0.60721, 0.87716, 0.84258],      
           [0.59608, 0.87292, 0.84357],      
           [0.58494, 0.86849, 0.84438],      
           [0.57379, 0.86386, 0.84502],      
           [0.56267, 0.85903, 0.84548],      
           [0.55159, 0.85402, 0.84576],      
           [0.54058, 0.84884, 0.84588],      
           [0.52966, 0.84347, 0.84582],      
           [0.51886, 0.83795, 0.8456],      
           [0.50819, 0.83227, 0.84522],      
           [0.49767, 0.82643, 0.84467],      
           [0.48733, 0.82046, 0.84397],      
           [0.47718, 0.81436, 0.84312],      
           [0.46725, 0.80814, 0.84213],      
           [0.45755, 0.8018, 0.84099],      
           [0.44809, 0.79537, 0.83973],      
           [0.43889, 0.78885, 0.83833],      
           [0.42997, 0.78225, 0.83681],      
           [0.42131, 0.77557, 0.83517],      
           [0.41296, 0.76883, 0.83343],      
           [0.40486, 0.76204, 0.83159],      
           [0.39707, 0.75521, 0.82964],      
           [0.38957, 0.74833, 0.82761],      
           [0.38235, 0.74142, 0.82549],      
           [0.37542, 0.7345, 0.8233],      
           [0.36877, 0.72754, 0.82104],      
           [0.36238, 0.72058, 0.8187],      
           [0.35627, 0.71361, 0.81632],      
           [0.3504, 0.70664, 0.81387],      
           [0.34477, 0.69966, 0.81138],      
           [0.33939, 0.69269, 0.80884],      
           [0.33422, 0.68572, 0.80626],      
           [0.32926, 0.67875, 0.80364],      
           [0.32448, 0.67181, 0.801],      
           [0.31992, 0.66486, 0.79832],      
           [0.31551, 0.65795, 0.79562],      
           [0.31127, 0.65104, 0.7929],      
           [0.30718, 0.64414, 0.79015],      
           [0.30322, 0.63727, 0.78739],      
           [0.29942, 0.63042, 0.78462],      
           [0.29571, 0.62358, 0.78184],      
           [0.29213, 0.61676, 0.77904],      
           [0.28864, 0.60995, 0.77624],      
           [0.28523, 0.60318, 0.77343],      
           [0.28193, 0.59642, 0.77063],      
           [0.2787, 0.58967, 0.76781],      
           [0.27554, 0.58296, 0.765],      
           [0.27241, 0.57628, 0.76218],      
           [0.26939, 0.56959, 0.75937],      
           [0.26638, 0.56295, 0.75656],      
           [0.26345, 0.5563, 0.75375],      
           [0.26053, 0.5497, 0.75095],      
           [0.25766, 0.54311, 0.74814],      
           [0.25486, 0.53655, 0.74534],      
           [0.25205, 0.53, 0.74255],      
           [0.24928, 0.52347, 0.73977],      
           [0.24654, 0.51697, 0.73698],      
           [0.24382, 0.51048, 0.73421],      
           [0.24114, 0.50402, 0.73143],      
           [0.23846, 0.49758, 0.72867],      
           [0.23583, 0.49117, 0.72592],      
           [0.23317, 0.48475, 0.72317],      
           [0.23056, 0.47838, 0.72043],      
           [0.22798, 0.47202, 0.7177],      
           [0.22538, 0.46567, 0.71496],      
           [0.22282, 0.45936, 0.71224],      
           [0.22026, 0.45306, 0.70953],      
           [0.2177, 0.44678, 0.70682],      
           [0.21514, 0.44051, 0.70412],      
           [0.21262, 0.43427, 0.70142],      
           [0.21009, 0.42806, 0.69874],      
           [0.20758, 0.42184, 0.69606],      
           [0.20507, 0.41566, 0.69339],      
           [0.20256, 0.4095, 0.69071],      
           [0.20005, 0.40335, 0.68806],      
           [0.19757, 0.3972, 0.6854],      
           [0.19509, 0.3911, 0.68275],      
           [0.19259, 0.38498, 0.6801],      
           [0.1901, 0.3789, 0.67748],      
           [0.18765, 0.37283, 0.67484],      
           [0.18515, 0.36678, 0.67222],      
           [0.18262, 0.36073, 0.66959],      
           [0.18013, 0.35472, 0.66697],      
           [0.17766, 0.3487, 0.66436],      
           [0.17513, 0.34271, 0.66176],      
           [0.17259, 0.33671, 0.65915],      
           [0.17007, 0.33072, 0.65656],      
           [0.16752, 0.32475, 0.65396],      
           [0.16494, 0.3188, 0.65137],      
           [0.16238, 0.31285, 0.64878],      
           [0.15974, 0.30691, 0.64619],      
           [0.15712, 0.30097, 0.64361],      
           [0.15446, 0.29504, 0.64103],      
           [0.15176, 0.28914, 0.63846],      
           [0.14904, 0.28322, 0.63589],      
           [0.14627, 0.2773, 0.63331],      
           [0.14346, 0.27138, 0.63075],      
           [0.1406, 0.26547, 0.62818],      
           [0.13769, 0.25957, 0.62561],      
           [0.1347, 0.25365, 0.62305],      
           [0.13163, 0.24774, 0.62049],      
           [0.12849, 0.24182, 0.61792],      
           [0.12528, 0.2359, 0.61535],      
           [0.12194, 0.22993, 0.61279],      
           [0.11859, 0.22399, 0.61023],      
           [0.11502, 0.21805, 0.60768],      
           [0.11142, 0.21209, 0.60511],      
           [0.10761, 0.20611, 0.60255],      
           [0.1037, 0.20006, 0.59999]]      
      
roma_map = LinearSegmentedColormap.from_list('roma', cm_data)      
# For use of "viscm view"      
test_cm = roma_map      
      
if __name__ == "__main__":      
    import matplotlib.pyplot as plt      
    import numpy as np      
      
    try:      
        from viscm import viscm      
        viscm(roma_map)      
    except ImportError:      
        print("viscm not found, falling back on simple display")      
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',      
                   cmap=roma_map)      
    plt.show()      
