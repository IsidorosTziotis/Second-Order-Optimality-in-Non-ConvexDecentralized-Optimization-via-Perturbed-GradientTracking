import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
restrict= 1500
nodes = 10
if __name__ == "__main__":
    file_to_load = "withnoise_1500it_10nodes.py_08-12-2020-11-41-18.p"
    experiment_directory_name = "experiment_outputs"
    with open("{}/{}".format(experiment_directory_name, file_to_load), 'rb') as f:
        saved_results = pickle.load(f)
    # then you can access the entries of the map like so:
    saved_results["loss"]
    normalized_loss = [i/nodes for i in saved_results["loss"]]
    saved_results["t1"]
    saved_results["t2"]
    saved_results["criterion"]
    saved_results["phases"]

    file_to_load = "nonoise_1500it_10nodes.py_08-12-2020-11-16-10.p"
    experiment_directory_name = "experiment_outputs"
    with open("{}/{}".format(experiment_directory_name, file_to_load), 'rb') as f:
        saved_results2 = pickle.load(f)
    # then you can access the entries of the map like so:
    saved_results2["loss"]
    normalized_loss2 = [i / nodes for i in saved_results2["loss"]]
    saved_results2["t1"]
    saved_results2["t2"]
    saved_results2["criterion"]
    saved_results2["phases"]

#print(saved_results["loss"])
#print(saved_results2["loss"])
#print(type(saved_results["loss"]))
#input("stop")

font = font_manager.FontProperties(family='Arial',
                                   weight='bold',
                                   style='normal', size=14)

plt.plot(range(restrict), normalized_loss[0:restrict], marker='', linewidth='3', label='D-GET')
plt.plot(range(restrict), normalized_loss2[0:restrict], marker='', linewidth='3', label='PDGT')

plt.grid(True)
plt.xlabel('Iterations',fontsize =15)
plt.ylabel('Average Loss', fontsize =15)
plt.title('Loss to Iterations - nodes=10/rank=20',fontsize=15)
plt.legend(prop = font)
#plt.xscale('log')
plt.savefig('Loss_to_Iterationss.png')
plt.show()
#plt.savefig("{}/{}.png".format(self.output_dir, self.plot_title))

plt.plot(range(restrict), saved_results["t1"][0:restrict], marker='', linewidth='3', label='D-GET')
plt.plot(range(restrict), saved_results2["t1"][0:restrict], marker='', linewidth='3', label='PDGT')

plt.grid(True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Iterations',fontsize = 15)
plt.ylabel('Squared Norm of Average Gradient ',fontsize = 15)
#plt.yscale('log')
plt.title('Grad to Iterations - nodes=10/rank=20',fontsize = 15)


plt.legend(prop=font)
# plt.xscale('log')
plt.savefig('Gradient_to_Iterations.png')
plt.show()

plt.plot(range(restrict), saved_results["t2"][0:restrict], marker='', linewidth='3', label='D-GET')
plt.plot(range(restrict), saved_results2["t2"][0:restrict], marker='', linewidth='3', label='PDGT')

plt.grid(True)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Iterations',fontsize = 15)
plt.ylabel('Consensus Error ',fontsize = 15)
#plt.yscale('log')
plt.title('Consensus to Iterations - nodes=10/rank=20',fontsize = 15)


plt.legend(prop=font)
# plt.xscale('log')
plt.savefig('Consensus_Error_to_Iterations.png')
plt.show()

