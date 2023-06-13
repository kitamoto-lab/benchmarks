import torch
import torchvision


def predict(model, images, num_frames_to_predict):
    batch_size, time, channels, height, width = images.size()

    model_in = images
    predicted_imgs = None
    for i in range(num_frames_to_predict):
    	model_out = model.forward(model_in)
    	model_out = model_out.unsqueeze(2) # add in the "time" channel
    	if predicted_imgs is None:
    		predicted_imgs = model_out
    	else:
    		predicted_imgs = torch.cat((predicted_imgs, model_out), dim=2)
    	model_in = torch.concat((model_in, model_out), dim=1)
    	model_in = model_in[:,1:] # remove first img
    predicted_imgs = predicted_imgs.reshape([batch_size, num_frames_to_predict, channels, height, width])
    return predicted_imgs


def make_prediction_video(images, savepath):
	fig, ax = plt.subplots()

	num_images, channels, height, width = images.size()
	images = images.reshape((num_images, height, width, channels))
	
	ims = []
	for i in range(len(images)):
		im = ax.imshow(images[i], animated=True)
		if i == 0:
			ax.imshow(images[i])
		ims.append([im])

	ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
	                                repeat_delay=1000)
	plt.savefig(savepath)
	plt.clear()


def create_grid(x, y_hat, y, prediction_start_hour, length_of_prediction):        
    # predictions with input for illustration purposes
    preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)[0]

    # entire input and ground truth
    y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)[0]

    # error (l2 norm) plot between pred and ground truth
    difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
    zeros = torch.zeros(difference.shape)
    difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[0]

    # concat all images
    final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

    num_prev = prediction_start_hour
    num_ahead = length_of_prediction
    # make them into a single grid image file
    grid = torchvision.utils.make_grid(final_image, nrow=num_prev+num_ahead)
    return grid

def read_validation_indices(filepath):
    with open(filepath, 'r') as f:
        line = f.readlines()[0][1:-1]
        indices_list = [int(num) for num in line.split(',')]
    return indices_list