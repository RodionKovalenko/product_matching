import pathlib
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

saved_model_evaluation_path = 'sbert_trained_model_sim_score\\eval'
current_abs_path = str(pathlib.Path().resolve())
model_evaluation_path = current_abs_path + '\\' + saved_model_evaluation_path

print (model_evaluation_path)
data = None


def plot_evaluation(data, save_image = False):       
    fig, ax = plt.subplots(1,1)   
    num_epochs = np.max(data['epoch'] + 1)
    y = sorted(data['cosine_pearson']) 
    z = sorted(data['cosine_spearman']) 
    x = np.arange (0, num_epochs, num_epochs/ len(y))        
    
    ax.set_title('SBERT-Modellevaluierung')
    ax.plot(x, y, x, z)
    ax.set_yticks(np.arange(0.6, 1, 0.05))
    ax.set_xlabel('Epoche')
    ax.set_ylabel('Korrelationskoeffizienten')
    ax.legend(['Bravais-Pearson', 'Spearman'])
    ax.grid(True)
    print(y)
    print(z)
    
    if save_image:
       plt.savefig('sbert-evaluation.png')
    
    plt.show()
    
   

if os.path.isdir(model_evaluation_path):
    print('model evaluation found')   
    data = pd.read_csv(model_evaluation_path + '\\similarity_evaluation_results.csv')
    plot_evaluation(data, save_image = True)
else:
    print('no model evaluation found')
    


