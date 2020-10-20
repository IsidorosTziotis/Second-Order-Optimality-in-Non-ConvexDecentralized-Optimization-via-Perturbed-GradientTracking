import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, plot_title, output_dir):
        self.plot_title = plot_title
        self.output_dir = output_dir

    def plot(self, loss_values, term_1_values, term_2_values, criterion_values, phases):
        # TODO: plot stuff
        plt.plot(range(1500), loss_values[0:1500], marker='', linestyle='-', label='With noise')
        plt.xlabel('EPOCHS')
        plt.ylabel('LOSS')
        plt.legend()
        #plt.xscale('log')
        #plt.savefig('Gradient_to_Machine_Time.png')
        plt.show()
        #plt.savefig("{}/{}.png".format(self.output_dir, self.plot_title))

        plt.plot(range(1500), term_1_values[0:1500], marker='', linestyle='-', label='With noise')
        plt.xlabel('EPOCHS')
        plt.ylabel('Gradient')
        plt.legend()
        # plt.xscale('log')
        # plt.savefig('Gradient_to_Machine_Time.png')
        plt.show()

        # plt.savefig("{}/{}.png".format(self.output_dir, self.plot_title))