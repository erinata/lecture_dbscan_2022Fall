import pandas
import numpy
import matplotlib.pyplot as pyplot

from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors



dataset = pandas.read_csv("dataset_moon.csv")

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_moon.png")
pyplot.close()



knn_machine = NearestNeighbors(n_neighbors=5)
results = knn_machine.fit(dataset)
distances, index = results.kneighbors(dataset)
distances = numpy.sort(distances, axis = 0)


distances = [sum(d)/4 for d in distances]

print(distances)

pyplot.plot(distances)
pyplot.savefig("knn_distance_moon.png")
pyplot.close()


dbscan_machine = DBSCAN(eps=0.125, min_samples=4)
results = dbscan_machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results)
pyplot.savefig("scatterplot_moon_with_color.png")
pyplot.close()





dataset = pandas.read_csv("dataset_new_moon.csv")

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_new_moon.png")
pyplot.close()


knn_machine = NearestNeighbors(n_neighbors=5)
results = knn_machine.fit(dataset)
distances, index = results.kneighbors(dataset)
distances = numpy.sort(distances, axis = 0)

distances = [sum(d)/4 for d in distances]

print(distances)

pyplot.plot(distances)
pyplot.savefig("knn_distance_new_moon.png")
pyplot.close()


dbscan_machine = DBSCAN(eps=0.14, min_samples=4)
results = dbscan_machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results)
pyplot.savefig("scatterplot_new_moon_with_color.png")
pyplot.close()



