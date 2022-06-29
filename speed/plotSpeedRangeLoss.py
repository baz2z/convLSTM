import matplotlib.pyplot as plt
import numpy as np




waveSpeed = np.arange(1, 5.2, step = 0.2)
lossBaseline = [10.994924, 7.758896, 5.5064387, 4.0678596, 3.0834432, 2.294477, 1.6719538, 1.1206443, 0.5998588, 0.17512695, 0.00012725494, 0.14791104, 0.4534192, 0.7467748, 0.96388376, 1.115495, 1.2506237, 1.3339688, 1.4201161, 1.483208, 1.5049362]
lossLateral1:1 = [11.185662, 8.10703, 6.086627, 4.2027154, 3.1262467, 2.2708888, 1.6302786, 1.1019275, 0.58252174, 0.17612262, 8.155341e-05, 0.15082759, 0.4457038, 0.77283895, 0.98777974, 1.2008533, 1.249622, 1.3026018, 1.3559281, 1.4009202, 1.3509963]
lossTwoLayer2:1 = [12.13248, 8.610303, 6.4476905, 4.8884563, 3.7583184, 2.8336864, 2.0041306, 1.32812, 0.7288767, 0.21524453, 0.0001554942, 0.19138703, 0.535287, 0.8450791, 1.0633775, 1.2018021, 1.3221457, 1.3691881, 1.416764, 1.406229, 1.3756577]
lossSkip2:1 = [10.573677, 7.4452615, 5.436146, 4.0172276, 3.0328472, 2.2440789, 1.6121804, 1.0821803, 0.58405256, 0.17379682, 2.2962651e-05, 0.1471434, 0.43162385, 0.7288162, 1.0213165, 1.2351208, 1.3774546, 1.4033244, 1.4384277, 1.4426041, 1.4076384]
lossDW1:2 = [10.817171, 7.5532775, 5.4848084, 4.0622015, 3.073295, 2.2925096, 1.6867669, 1.1456776, 0.60942715, 0.17213121, 8.371836e-05, 0.14308532, 0.44341436, 0.7139676, 0.89892226, 0.9930555, 1.068342, 1.0763566, 1.0976965, 1.1131442, 1.1076001]


plt.scatter(waveSpeed, lossBaseline, label="Baseline", c="blue")
plt.scatter(waveSpeed, lossLateral1, label="Lateral", c="red")
plt.scatter(waveSpeed, lossTwoLayer2, label="TwoLayer", c="green")
plt.scatter(waveSpeed, lossSkip2, label="Skip", c="pink")
plt.scatter(waveSpeed, lossDW1, label="DepthWise", c="brown")

linex = np.arange(1, 6)
liney = np.repeat(lossBaseline[10] * 2, 5)
plt.plot(linex, liney, 'k-', c = 'blue')
liney = np.repeat(lossLateral1[10] * 2, 5)
plt.plot(linex, liney, 'k-', c = 'red')
liney = np.repeat(lossTwoLayer2[10] * 2, 5)
plt.plot(linex, liney, 'k-', c = 'green')
liney = np.repeat(lossSkip2[10] * 2, 5)
plt.plot(linex, liney, 'k-', c = 'pink')
liney = np.repeat(lossDW1[10] * 2, 5)
plt.plot(linex, liney, 'k-', c = 'brown')

plt.xlabel("waveSpeed")
plt.ylabel("loss")
plt.legend()
#plt.ylim([-0.2, 2])
plt.yscale("log")
plt.savefig("speedRangeLossLog.png")
plt.show()