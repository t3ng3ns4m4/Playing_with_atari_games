import matplotlib.pyplot as plt

def display_observation_image(observation):
    observation = observation.squeeze(0)
    observation = observation.squeeze(0)
    plt.imshow(observation, cmap='gray')  
    plt.axis('off')  
    plt.show()