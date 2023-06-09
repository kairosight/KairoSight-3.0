# 
#         tokyo
#                   www.fabiocrameri.ch/visualisation
from matplotlib.colors import LinearSegmentedColormap      
      
cm_data = [[0.10387, 0.056805, 0.20243],      
           [0.10975, 0.059104, 0.20564],      
           [0.11566, 0.061046, 0.20884],      
           [0.1216, 0.063055, 0.21209],      
           [0.12757, 0.064935, 0.21531],      
           [0.13351, 0.066895, 0.21858],      
           [0.13939, 0.068939, 0.22184],      
           [0.14535, 0.070894, 0.22512],      
           [0.15119, 0.072938, 0.22845],      
           [0.15708, 0.07497, 0.23177],      
           [0.16298, 0.077038, 0.23511],      
           [0.16886, 0.079188, 0.23844],      
           [0.17475, 0.081435, 0.24182],      
           [0.18061, 0.083639, 0.24519],      
           [0.18652, 0.085843, 0.24863],      
           [0.19242, 0.088221, 0.25206],      
           [0.19833, 0.090586, 0.2555],      
           [0.20429, 0.092949, 0.25898],      
           [0.21021, 0.095479, 0.26246],      
           [0.21619, 0.098018, 0.26598],      
           [0.22212, 0.10056, 0.26951],      
           [0.22812, 0.10325, 0.27306],      
           [0.2341, 0.10596, 0.27664],      
           [0.24009, 0.10871, 0.28021],      
           [0.2461, 0.11159, 0.28383],      
           [0.25214, 0.11445, 0.28745],      
           [0.25815, 0.11746, 0.29112],      
           [0.26421, 0.12052, 0.29477],      
           [0.27026, 0.12365, 0.29845],      
           [0.27631, 0.12687, 0.30215],      
           [0.28236, 0.13021, 0.30588],      
           [0.28839, 0.13354, 0.3096],      
           [0.29442, 0.13697, 0.31333],      
           [0.30046, 0.14049, 0.31707],      
           [0.30651, 0.14408, 0.32083],      
           [0.3125, 0.14775, 0.32457],      
           [0.3185, 0.15147, 0.32834],      
           [0.32447, 0.15529, 0.33211],      
           [0.33042, 0.15918, 0.33586],      
           [0.33634, 0.16315, 0.33963],      
           [0.34223, 0.16717, 0.34337],      
           [0.34808, 0.17124, 0.34711],      
           [0.35387, 0.1754, 0.35084],      
           [0.35963, 0.17962, 0.35455],      
           [0.36533, 0.18389, 0.35824],      
           [0.37097, 0.18825, 0.36192],      
           [0.37657, 0.19261, 0.36557],      
           [0.38208, 0.19706, 0.3692],      
           [0.38753, 0.2015, 0.37279],      
           [0.3929, 0.20606, 0.37638],      
           [0.3982, 0.21059, 0.3799],      
           [0.40342, 0.21518, 0.3834],      
           [0.40854, 0.21982, 0.38685],      
           [0.41357, 0.22443, 0.39027],      
           [0.41851, 0.22913, 0.39364],      
           [0.42335, 0.2338, 0.39696],      
           [0.42811, 0.2385, 0.40025],      
           [0.43275, 0.24322, 0.40348],      
           [0.43729, 0.24796, 0.40666],      
           [0.44172, 0.25267, 0.40979],      
           [0.44603, 0.25739, 0.41286],      
           [0.45026, 0.26211, 0.41587],      
           [0.45436, 0.26682, 0.41882],      
           [0.45835, 0.27152, 0.42172],      
           [0.46223, 0.27624, 0.42457],      
           [0.466, 0.28088, 0.42737],      
           [0.46968, 0.28555, 0.43009],      
           [0.47321, 0.29019, 0.43276],      
           [0.47666, 0.29481, 0.43537],      
           [0.48, 0.29942, 0.43793],      
           [0.48321, 0.30398, 0.44041],      
           [0.48633, 0.30854, 0.44286],      
           [0.48934, 0.31305, 0.44524],      
           [0.49226, 0.31754, 0.44756],      
           [0.49507, 0.322, 0.44984],      
           [0.49779, 0.32642, 0.45206],      
           [0.5004, 0.33082, 0.45422],      
           [0.50292, 0.33521, 0.45633],      
           [0.50535, 0.33954, 0.45839],      
           [0.50771, 0.34383, 0.4604],      
           [0.50996, 0.34812, 0.46237],      
           [0.51213, 0.35236, 0.46431],      
           [0.51424, 0.35658, 0.46617],      
           [0.51624, 0.36075, 0.468],      
           [0.51819, 0.36491, 0.4698],      
           [0.52005, 0.36904, 0.47155],      
           [0.52185, 0.37313, 0.47324],      
           [0.52359, 0.37721, 0.47493],      
           [0.52526, 0.38125, 0.47656],      
           [0.52687, 0.38525, 0.47815],      
           [0.52841, 0.38925, 0.47973],      
           [0.5299, 0.39321, 0.48125],      
           [0.53134, 0.39714, 0.48276],      
           [0.53273, 0.40107, 0.48423],      
           [0.53406, 0.40495, 0.48567],      
           [0.53535, 0.40883, 0.4871],      
           [0.53661, 0.41269, 0.48849],      
           [0.53781, 0.41652, 0.48985],      
           [0.53897, 0.42033, 0.49121],      
           [0.54009, 0.42412, 0.49252],      
           [0.54118, 0.42791, 0.49383],      
           [0.54224, 0.43167, 0.49511],      
           [0.54326, 0.43542, 0.49637],      
           [0.54425, 0.43914, 0.49763],      
           [0.54522, 0.44286, 0.49885],      
           [0.54616, 0.44657, 0.50006],      
           [0.54706, 0.45026, 0.50126],      
           [0.54795, 0.45394, 0.50244],      
           [0.54882, 0.45761, 0.50361],      
           [0.54965, 0.46127, 0.50477],      
           [0.55048, 0.46492, 0.50591],      
           [0.55128, 0.46856, 0.50705],      
           [0.55207, 0.4722, 0.50818],      
           [0.55283, 0.47583, 0.50929],      
           [0.55358, 0.47945, 0.51039],      
           [0.55433, 0.48306, 0.51148],      
           [0.55505, 0.48667, 0.51257],      
           [0.55576, 0.49027, 0.51365],      
           [0.55646, 0.49387, 0.51472],      
           [0.55716, 0.49747, 0.51578],      
           [0.55783, 0.50106, 0.51684],      
           [0.5585, 0.50465, 0.5179],      
           [0.55917, 0.50823, 0.51895],      
           [0.55983, 0.5118, 0.51997],      
           [0.56047, 0.51538, 0.52102],      
           [0.56111, 0.51897, 0.52204],      
           [0.56175, 0.52253, 0.52308],      
           [0.56239, 0.52611, 0.5241],      
           [0.56301, 0.52969, 0.52511],      
           [0.56363, 0.53325, 0.52613],      
           [0.56425, 0.53684, 0.52714],      
           [0.56485, 0.5404, 0.52815],      
           [0.56546, 0.54398, 0.52916],      
           [0.56608, 0.54755, 0.53016],      
           [0.56669, 0.55112, 0.53116],      
           [0.56729, 0.5547, 0.53216],      
           [0.56789, 0.55827, 0.53315],      
           [0.5685, 0.56186, 0.53414],      
           [0.5691, 0.56543, 0.53513],      
           [0.5697, 0.56901, 0.53612],      
           [0.57031, 0.57261, 0.53711],      
           [0.57092, 0.5762, 0.5381],      
           [0.57153, 0.57978, 0.53908],      
           [0.57214, 0.58337, 0.54006],      
           [0.57276, 0.58696, 0.54104],      
           [0.57337, 0.59056, 0.54202],      
           [0.57399, 0.59417, 0.54301],      
           [0.57461, 0.59778, 0.54399],      
           [0.57524, 0.60139, 0.54497],      
           [0.57589, 0.60501, 0.54596],      
           [0.57653, 0.60864, 0.54694],      
           [0.57717, 0.61225, 0.54792],      
           [0.57782, 0.61589, 0.54891],      
           [0.57849, 0.61954, 0.54989],      
           [0.57918, 0.62318, 0.55087],      
           [0.57987, 0.62683, 0.55187],      
           [0.58056, 0.6305, 0.55286],      
           [0.58127, 0.63417, 0.55385],      
           [0.582, 0.63785, 0.55485],      
           [0.58274, 0.64153, 0.55585],      
           [0.5835, 0.64523, 0.55687],      
           [0.58428, 0.64895, 0.55789],      
           [0.58508, 0.65267, 0.55891],      
           [0.5859, 0.65641, 0.55995],      
           [0.58674, 0.66015, 0.56098],      
           [0.5876, 0.66392, 0.56204],      
           [0.5885, 0.6677, 0.5631],      
           [0.58942, 0.67149, 0.56418],      
           [0.59039, 0.67531, 0.56526],      
           [0.59139, 0.67914, 0.56637],      
           [0.59242, 0.683, 0.56749],      
           [0.59349, 0.68688, 0.56863],      
           [0.59461, 0.69078, 0.56979],      
           [0.59578, 0.6947, 0.57097],      
           [0.597, 0.69866, 0.57219],      
           [0.59826, 0.70264, 0.57342],      
           [0.5996, 0.70666, 0.57468],      
           [0.60099, 0.7107, 0.57599],      
           [0.60246, 0.71478, 0.57731],      
           [0.604, 0.7189, 0.57868],      
           [0.6056, 0.72305, 0.5801],      
           [0.60731, 0.72725, 0.58155],      
           [0.60909, 0.73148, 0.58305],      
           [0.61097, 0.73576, 0.5846],      
           [0.61294, 0.7401, 0.58621],      
           [0.61503, 0.74447, 0.58787],      
           [0.61723, 0.7489, 0.58959],      
           [0.61955, 0.75338, 0.59139],      
           [0.62198, 0.75792, 0.59325],      
           [0.62456, 0.76251, 0.59518],      
           [0.62727, 0.76716, 0.5972],      
           [0.63013, 0.77187, 0.5993],      
           [0.63314, 0.77664, 0.60148],      
           [0.6363, 0.78146, 0.60376],      
           [0.63963, 0.78635, 0.60611],      
           [0.64313, 0.79129, 0.60859],      
           [0.64681, 0.79629, 0.61116],      
           [0.65067, 0.80136, 0.61382],      
           [0.6547, 0.80648, 0.61662],      
           [0.65894, 0.81164, 0.61951],      
           [0.66337, 0.81686, 0.62251],      
           [0.66798, 0.82213, 0.62564],      
           [0.6728, 0.82743, 0.62888],      
           [0.6778, 0.83276, 0.63223],      
           [0.683, 0.83813, 0.6357],      
           [0.68839, 0.84351, 0.63929],      
           [0.69397, 0.84891, 0.64299],      
           [0.69973, 0.85431, 0.64681],      
           [0.70567, 0.85972, 0.65072],      
           [0.71178, 0.86509, 0.65474],      
           [0.71804, 0.87045, 0.65887],      
           [0.72445, 0.87578, 0.66309],      
           [0.731, 0.88106, 0.66739],      
           [0.73768, 0.88628, 0.67178],      
           [0.74447, 0.89143, 0.67624],      
           [0.75136, 0.89651, 0.68075],      
           [0.75833, 0.9015, 0.68534],      
           [0.76537, 0.90639, 0.68996],      
           [0.77246, 0.91117, 0.69462],      
           [0.77959, 0.91583, 0.69932],      
           [0.78675, 0.92037, 0.70404],      
           [0.79392, 0.92477, 0.70878],      
           [0.80108, 0.92904, 0.71351],      
           [0.80822, 0.93317, 0.71825],      
           [0.81534, 0.93716, 0.72297],      
           [0.82241, 0.94098, 0.72768],      
           [0.82944, 0.94466, 0.73237],      
           [0.8364, 0.94818, 0.73701],      
           [0.84329, 0.95154, 0.74163],      
           [0.8501, 0.95476, 0.74621],      
           [0.85683, 0.95782, 0.75075],      
           [0.86348, 0.96072, 0.75524],      
           [0.87002, 0.96348, 0.75967],      
           [0.87648, 0.96609, 0.76406],      
           [0.88284, 0.96857, 0.76839],      
           [0.8891, 0.9709, 0.77268],      
           [0.89526, 0.97311, 0.77691],      
           [0.90133, 0.97518, 0.78108],      
           [0.90729, 0.97714, 0.7852],      
           [0.91316, 0.97898, 0.78927],      
           [0.91894, 0.98071, 0.79329],      
           [0.92463, 0.98234, 0.79725],      
           [0.93023, 0.98386, 0.80117],      
           [0.93575, 0.9853, 0.80504],      
           [0.94119, 0.98664, 0.80888],      
           [0.94656, 0.98791, 0.81266],      
           [0.95185, 0.9891, 0.81642],      
           [0.95708, 0.99022, 0.82012],      
           [0.96225, 0.99128, 0.82381],      
           [0.96736, 0.99228, 0.82746],      
           [0.97242, 0.99323, 0.83108],      
           [0.97743, 0.99412, 0.83468],      
           [0.9824, 0.99498, 0.83825],      
           [0.98733, 0.99579, 0.84181],      
           [0.99222, 0.99658, 0.84535],      
           [0.99708, 0.99733, 0.84887]]      
      
tokyo_map = LinearSegmentedColormap.from_list('tokyo', cm_data)      
# For use of "viscm view"      
test_cm = tokyo_map      
      
if __name__ == "__main__":      
    import matplotlib.pyplot as plt      
    import numpy as np      
      
    try:      
        from viscm import viscm      
        viscm(tokyo_map)      
    except ImportError:      
        print("viscm not found, falling back on simple display")      
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',      
                   cmap=tokyo_map)      
    plt.show()      
