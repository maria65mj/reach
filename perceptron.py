import numpy as np 
import matplotlib.pyplot as plt

def read_csv(filename, header = True, has_target = True):
	with open(filename, "r") as f:
		if (header):
			header = f.readline().split(",")
		data_list = []
		target = []
		for l in f.readlines():
			line = l.split(",")
			if(has_target):
				target.append(float(line.pop(0)))
			line = [float(x) for x in line] #now convert all to numeric
			line.append(1.0) #Bias feature
			data_list.append(line)
		target = [1 if x == 3 else -1 for x in target]
		if(not has_target):
			return(np.array(data_list))
		else:
			return(np.array(data_list), np.array(target))

def perceptron(train_matrix, train_target, dev_matrix, dev_target, iters, average = True):
	print("Training online perceptron clasifier with ", iters, " iterations.")
	w = np.zeros(np.size(train_matrix, axis = 1))
	w_avg = np.zeros(np.size(train_matrix, axis = 1))
	s = 1
	train_accuracies = []
	dev_accuracies = []
	weights_list = []
	for i in range(iters):
		for j in range(np.size(train_matrix, axis = 0)):
			if train_target[j] * np.matmul(w, train_matrix[j]) <= 0:
				w = w + train_target[j] * train_matrix[j]
			if(average):
				w_avg = (s * w_avg + w) / (s + 1)
				s = s + 1
		if(average): 
			train_accuracies.append(accuracy(predict(train_matrix, w_avg), train_target)) #train_accuracy
			dev_accuracies.append(accuracy(predict(dev_matrix, w_avg), dev_target)) #validation accuracy
			weights_list.append(w_avg)
		else:
			train_accuracies.append(accuracy(predict(train_matrix, w), train_target)) #train_accuracy
			dev_accuracies.append(accuracy(predict(dev_matrix, w), dev_target)) #validation accuracy
			weights_list.append(w)
	return(weights_list, train_accuracies, dev_accuracies)

def kernel_perceptron(train_matrix, train_target, dev_matrix, dev_target, iters, p):
	print("Training kernel perceptron clasifier with ", iters, " iterations.")
	train_accuracies = []
	dev_accuracies = []
	weights_list = []
	alpha = np.zeros(np.size(train_matrix, axis = 0))
	w = np.zeros(np.size(train_matrix, axis = 1))
	gram = gram_matrix(train_matrix, train_matrix, p)
	for i in range(iters):
		for j in range(np.size(train_matrix, axis = 1)):
			u = np.sum( alpha * gram[j] * train_target)
			if train_target[j]*u <= 0:
				alpha[j] = alpha[j] + 1
		train_accuracies.append(accuracy(kernel_predict(train_matrix, train_target, train_matrix, alpha, p), train_target)) #train_accuracy
		dev_accuracies.append(accuracy(kernel_predict(train_matrix, train_target, dev_matrix, alpha, p), dev_target)) #validation accuracy
		weights_list.append(alpha)
	return(weights_list, train_accuracies, dev_accuracies)

def predict(data_matrix, weights):
	preds = np.matmul(data_matrix, weights)
	preds = [1 if x >=0 else -1 for x in preds]
	return(preds)

def kernel_predict(train_matrix, train_target, data_matrix, alpha, order):
	gram_new = gram_matrix(train_matrix, data_matrix, order)
	y_pred = np.zeros(np.size(data_matrix, 0))
	for i in range(np.size(data_matrix, 0)):
		y_pred[i] = np.sign(np.sum(alpha*train_target*gram_new[:,i]))
	return(y_pred)


def accuracy(preds, true):
	return(sum(preds == true)/ len(true))


def poly_kernel(x1, x2, p):
	return((1 + np.matmul(x1, x2))**p)

def gram_matrix(train_matrix, data_matrix, order):
	equal = np.array_equal(data_matrix, train_matrix)
	gram = np.zeros((np.size(train_matrix, 0), np.size(data_matrix, 0)))
	for i in range(np.size(train_matrix, 0)):
		for j in range(i, np.size(data_matrix, 0)):
			k1 = poly_kernel(train_matrix[i], data_matrix[j], order)
			gram[i, j] = k1
			if equal or i == j:
				gram[j, i] = k1
			elif i <= np.size(data_matrix,0):
				k2 = poly_kernel(train_matrix[j], data_matrix[i], order)
				gram[j,i] = k2
			
	return(gram)

def write_items(filename, item_list):
	'''
	writes items to file for storage, one per line
	'''
	with open(filename, 'w+') as f:
		for item in item_list:
			f.write(str(item))
			f.write("\n")

def main():
	train_matrix, train_target = read_csv("pa2_train.csv", header = False)
	dev_matrix, dev_target = read_csv("pa2_valid.csv", header = False)
	print(np.shape(dev_matrix))
	test_matrix = read_csv("pa2_test_no_label.csv", header = False, has_target = False)

	#Online Perceptron
	all_weights, train_accuracies, dev_accuracies = perceptron(train_matrix, train_target, dev_matrix, dev_target, iters = 15, average = False)
	best_iters = np.argmax(dev_accuracies)
	best_accuracy = dev_accuracies[best_iters]
	best_weights = all_weights[best_iters]
	plt.plot(range(len(train_accuracies)), train_accuracies, color='skyblue', label = 'training')
	plt.plot(range(len(train_accuracies)), dev_accuracies, color='red', label = 'validation')
	plt.ylabel("accuracy")
	plt.xlabel("iters")
	plt.legend()
	plt.savefig("online_perceptron.png")
	plt.clf()

	print("Online perceptron best dev accuracy: ", best_accuracy)
	print("Online perceptron best iters: ", best_iters)

	online_predictions = predict(test_matrix, best_weights)
	write_items("oplabel.csv", online_predictions)

	#Average Perceptron
	all_weights, train_accuracies, dev_accuracies = perceptron(train_matrix, train_target, dev_matrix, dev_target, iters = 15, average = True)
	best_iters = np.argmax(dev_accuracies)
	best_accuracy = dev_accuracies[best_iters]
	best_weights = all_weights[best_iters]
	plt.plot(range(len(train_accuracies)), train_accuracies, color='skyblue', label = 'training')
	plt.plot(range(len(train_accuracies)), dev_accuracies, color='red', label = 'validation')
	plt.ylabel("accuracy")
	plt.xlabel("iters")
	plt.legend()
	plt.savefig("average_perceptron.png")
	plt.clf()
	print("Average perceptron best dev accuracy: ", best_accuracy)
	print("Average perceptron best iters: ", best_iters)

	average_predictions = predict(test_matrix, best_weights)
	write_items("aplabel.csv", average_predictions)

	#Kernel Perceptron
	best_p = 1
	overall_best_acuracy = 0
	overall_best_iters = 0
	overall_best_weights = None
	best_accurary_list = []
	for p in [1,2,3,4,5]:
		all_weights, train_accuracies, dev_accuracies = kernel_perceptron(train_matrix, train_target, dev_matrix, dev_target, iters = 15, p = p)
		best_iters = np.argmax(dev_accuracies)
		best_accuracy = dev_accuracies[best_iters]
		best_weights = all_weights[best_iters]
		best_accurary_list.append(best_accuracy)
		print(np.max(train_accuracies))
		plt.plot(range(len(train_accuracies)), train_accuracies, color='skyblue', label = 'training')
		plt.plot(range(len(train_accuracies)), dev_accuracies, color='red', label = 'validation')
		plt.ylabel("accuracy")
		plt.xlabel("iters")
		plt.legend()
		plt.savefig("".join(["kernel_perceptron", str(p), ".png"]))
		plt.clf()
		if overall_best_acuracy < best_accuracy:
			overall_best_acuracy = best_accuracy
			overall_best_iters = best_iters
			overall_best_weights = best_weights
			best_p = p
	
	plt.plot([1,2,3,4,5], best_accurary_list, color='skyblue')
	plt.ylabel("accuracy")
	plt.xlabel("p")
	plt.legend()
	plt.savefig("".join(["kernel_perceptron_all", ".png"]))
	plt.clf()
	print("Kernel perceptron best dev accuracy: ", best_accuracy)
	print("Kernel perceptron best order: ", best_p)
	print("Kernel perceptron best iters: ", best_iters)

	kernel_predictions = kernel_predict(train_matrix, train_target, test_matrix, best_weights, order = best_p)
	write_items("kplabel.csv", kernel_predictions)


if __name__ == '__main__':
	main()