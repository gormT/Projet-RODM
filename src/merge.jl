using Plots

using LinearAlgebra

using Clustering

include("struct/distance.jl")
include("utilities.jl")


"""
Essaie de regrouper des données en commençant par celles qui sont les plus proches.
Deux clusters de données peuvent être fusionnés en un cluster C s'il n'existe aucune données x_i pour aucune caractéristique j qui intersecte l'intervalle représenté par les bornes minimale et maximale de C pour j (x_i,j n'appartient pas à [min_{x_k dans C} x_k,j ; max_{k dans C} x_k,j]).

Entrées :
- x : caractéristiques des données d'entraînement
- y : classe des données d'entraînement
- percentage : le nombre de clusters obtenu sera égal à n * percentage
 
Sorties :
- un tableau de Cluster constituant une partition de x
"""
function exactMerge(x, y)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)
    
    # Pour chaque distance
    for distance in distances

        # Si les deux données associées ne sont pas déjà dans le même cluster
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        if cId1 != cId2
            c1 = clusters[cId1]
            c2 = clusters[cId2]

            # Si leurs clusters satisfont les conditions de fusion
            if canMerge(c1, c2, x, y)

                # Les fusionner
                merge!(c1, c2)
                for id in c2.dataIds
                    clusterId[id]= cId1
                end

                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end 
        end 
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

"""
Regroupe des données en commençant par celles qui sont les plus proches jusqu'à ce qu'un certain pourcentage de clusters soit atteint

Entrées :
- x : caractéristiques des données
- y : classe des données
- gamma : le regroupement se termine quand il reste un nombre de clusters < n * gamma ou que plus aucun regroupement n'est possible

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function simpleMerge(x, y, gamma)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)

    remainingClusters = n
    distanceId = 1

    # Pour chaque distance et tant que le nombre de cluster souhaité n'est pas atteint
    while distanceId <= length(distances) && remainingClusters > n * gamma

        distance = distances[distanceId]
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        # Si les deux données associées ne sont pas déjà dans le même cluster
        if cId1 != cId2
            remainingClusters -= 1

            # Fusionner leurs clusters 
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

            # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end 

"""
Test si deux clusters peuvent être fusionnés tout en garantissant l'optimalité

Entrées :
- c1 : premier cluster
- c2 : second cluster
- x  : caractéristiques des données d'entraînement
- y  : classe des données d'entraînement

Sorties :
- vrai si la fusion est possible ; faux sinon.
"""
function canMerge(c1::Cluster, c2::Cluster, x::Matrix{Float64}, y::Vector{Int})

    # Calcul des bornes inférieures si c1 et c2 étaient fusionnés
    mergedLBounds = min.(c1.lBounds, c2.lBounds)
    
    # Calcul des bornes supérieures si c1 et c2 étaient fusionnés
    mergedUBounds = max.(c1.uBounds, c2.uBounds)

    n = size(x, 1)
    id = 1
    canMerge = true

    # Tant que l'ont a pas vérifié que toutes les données n'intersectent la fusion de c1 et c2 sur aucune feature
    while id <= n && canMerge

        data = x[id, :]

        # Si la donnée n'est pas dans c1 ou c2 mais intersecte la fusion de c1 et c2 sur au moins une feature
        if !(id in c1.dataIds) && !(id in c2.dataIds) && isInABound(data, mergedLBounds, mergedUBounds)
            canMerge = false
        end 
        
        id += 1
    end 

    return canMerge
end

"""
Test si une donnée intersecte des bornes pour au moins une caractéristique 

Entrées :
- v : les caractéristique de la donnée
- lowerBounds : bornes inférieures pour chaque caractéristique
- upperBounds : bornes supérieures pour chaque caractéristique

Sorties :
- vrai s'il y a intersection ; faux sinon.
"""
function isInABound(v::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64})
    isInBound = false

    featureId = 1

    # Tant que toutes les features n'ont pas été testées et qu'aucune intersection n'a été trouvée
    while !isInBound && featureId <= length(v)

        # S'il y a intersection
        if v[featureId] >= lowerBounds[featureId] && v[featureId] <= upperBounds[featureId]
            isInBound = true
        end 
        featureId += 1
    end 

    return isInBound
end

"""
Fusionne deux clusters

Entrées :
- c1 : premier cluster
- c2 : second cluster

Sorties :
- aucune, c'est le cluster en premier argument qui contiendra le second
"""
function merge!(c1::Cluster, c2::Cluster)

    append!(c1.dataIds, c2.dataIds)
    c1.x = vcat(c1.x, c2.x)
    c1.lBounds = min.(c1.lBounds, c2.lBounds)
    c1.uBounds = max.(c1.uBounds, c2.uBounds)    
end


"""
Regroupe des données par la méthode DBSCAN

Entrées :
- x : caractéristiques des données
- y : classe des données
- minPts : le nombre minimum de points (seuil) proches les uns des autres pour qu'une région soit considérée "dense"
- eps : la distance utilisée pour définir que deux points sont proches l'un de l'autre

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function dbscanMerge(x, y, minPts, eps)
    n = length(y)           # nb de classes
    m = length(x[1,:])      # nb d'attributs pour chaque donnée
    p = size(x, 1)          # nb de points dans x

    corePts = zeros(Int, p)     # tableau des core points, = 1 si core point, = 0 sinon

    voisins = zeros(Int, p, p)  # voisins[i,j] = voisins[j,i] = 1 ssi i et j voisins

    for pt in 1:p   # on parcours tous les points
        nbVoisins = 0   # on initialise le compteur de voisins
        for i in pt+1:p    # on parcours tous les points tels que le couple (pt,i) n'ai pas été traité et on exclu les couples (pt,pt)
            dist = euclidean(x[i,:], x[pt,:])   # distance du point i au point pt
            if dist <= eps      # si distance inférieure à eps
                voisins[pt, i] = 1      # pt et i sont voisins
                voisins[i, pt] = 1      # et dans l'autre sens aussi
                nbVoisins = nbVoisins + 1   # on incrémente le compteur de voisins
            end
        end
        if nbVoisins >= minPts  # si pt a au moins minPts voisins, c'est un core point
            corePts[pt] = 1
        end
    end
    println("core points : $corePts")
    # println("matrice des voisins : $voisins")
    
    println("trace matrice des voisins : $(tr(voisins))")


    visited = zeros(Int, p)     # tableau des points visités, tout à zéro initialement

    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:p
        push!(clusters, Cluster(dataId, x, y))
    end


    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:p)

    for i in 1:p        # pour tout point i
        if visited[i] == 0 && corePts[i] == 1    # si i est un core point non visité
            visited[i] = 1      # on visite donc i
            println("on visite le point $i : $(x[i,:])")
            cId0 = clusterId[i]     # id du cluster contenant i
            println("$i est dans le cluster $cId0")
            c0 = clusters[cId0]    # i sera le point initial du cluster courant
            println("c0.x : $(c0.x)")
            println("c0.class : $(c0.class)")
            S = Vector{Int}([])     # S est l'ensemble des points qui seront dans le cluster à partir de cId0
            println("S = $S")
            for j in 1:p    # on visite tous les autres points
                if corePts[j] == 1 && visited[j] == 0 && voisins[i,j] == 1
                    push!(S, j)     # si j est un core point voisin de i, encore non visité, on l'ajoute à S
                    # S contient tous les core point voisins de i
                    visited[j] = 1
                end
            end
            while !isempty(S)       # on s'arrête lorsqu'il n'y a plus de voisins à visiter
                t = popfirst!(S)    # on prend le (premier) point t dans S et on le retire de S
                cId1 = clusterId[t]   # id du cluster contenant t
                c1 = clusters[cId1]    # cluster contenant t
                visited[t] = 1      # on visite donc t (normalement t a déjà été marqué comme visité mais bon...)
                
                # Si les deux données associées ne sont pas déjà dans le même cluster
                if cId0 != cId1
                    merge!(c0,c1)       # on fusionne les deux clusters
                    for id in c1.dataIds
                        clusterId[id]= cId0     # le cluster cId0 contient les points qui étaient dans c1
                    end
                    empty!(clusters[cId1].dataIds)     # on vide le second cluster
                end
                for l in 1:p
                    if visited[l] == 0 && voisins[t,l] == 1    # on parcourt les voisins non visités l de t
                        if corePts[l] == 1
                            push!(S, l)     # si l est un core point on l'ajoute à S
                            visited[l] = 1
                        else
                            cId2 = clusterId[l]
                            c2 = clusters[cId2]
                            visited[l] = 1     # on visite l puisqu'on l'ajoute au cluster immédiatement sans l'ajouter à S
                            # Si les deux données associées ne sont pas déjà dans le même cluster
                            if cId0 != cId2
                                merge!(c0,c2)   # on fusionne le cluster courant c0 avec le voisin l de t qui n'est pas un core point
                                for id in c2.dataIds
                                    clusterId[id]= cId0     # le cluster cId0 contient les points qui étaient dans c2
                                end
                                empty!(clusters[cId2].dataIds)     # on vide le second cluster
                            end
                        end
                    end
                end
            end
        end
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

"""
Retourne le nombre de voisins (points à une distance <= eps) du point pt dans l'ensemble de points x

Entrées :
- x : ensemble de points
- pt : id du point à considérer
- eps : la distance utilisée pour définir que deux points sont proches l'un de l'autre

Sorties :
- nbVoisins : le nombre de voisins de pt
"""

function cmbVoisins(x, pt, eps)
    nbVoisins = 0
    for i in 1:size(x,1)    # on parcours tous les points
        dist = euclidean(x[i,:], x[pt,:])   # distance du point i au point pt
        if dist <= eps
            nbVoisins= nbVoisins + 1
        end
    end
    return nbVoisins-1  # pt est à une distance 0 de lui-même, il faut le retirer des voisins
end



"""
Fonction de test de dbscanMerge
Affiche le clustering

"""
function test(dataSetName; minPts = 10, eps = 0.12)
    # Préparation des données
    include("../data/" * dataSetName * ".txt") 
    train, test = train_test_indexes(length(Y))
    X_train = X[train,:]
    Y_train = Y[train]
    X_test = X[test,:]
    Y_test = Y[test]

    p = size(X_train, 1)
    clusters = dbscanMerge(X_train, Y_train, minPts, eps)
    nb_clusters = size(clusters, 1)

    cluster_classes = zeros(Int, p)
    class_nb = 1
    for c in clusters
        dataIds = c.dataIds
        for id in dataIds
            cluster_classes[id] = class_nb
        end
        class_nb = class_nb + 1
    end
    println("$nb_clusters clusters created")
    
    title_name = "prnn_eps=" * string(eps) * "_minPts=" * string(minPts)

    scatter(X_train[:,1], X_train[:,2],
            xlabel = "X",
            ylabel = "Y",
            title = title_name,
            group = cluster_classes,
            legend = false,
            dpi = 1000)
end


# x caractéristiques des données, y classe des données, k nombre de clusters voulus
function KmeansMerge(x, y, k)

    # n = length(y)
    # m = length(x[1,:])
    n, m = size(x)[1], size(x)[2]
    
    clusters = Vector{Cluster}([])

    # On prend k centroides initiaux aléatoires parmi les données
    new_centroides = []
    possible_centroids = Vector(1:n)

    for i in 1:k
        if length(possible_centroids) == 0
            break
        end
        rand_index = rand(1:length(possible_centroids))
        push!(new_centroides, rand_index)
        deleteat!(possible_centroids, rand_index)
    end


    centroides = Vector([0 for i in 1:length(new_centroides)])

    # L'algorithme kmean se termine lorsque toutes les données sont partitionnés de manière stable, ie qu'ils ne changent plus de cluster 
    while new_centroides != centroides
        
        # Les anciens centroides deviennent ceux de l'itération précédente
        centroides = new_centroides

        clusters = Vector{Cluster}([])

        # Au début, créée un cluster par élément
        for data in 1:n
            push!(clusters, Cluster(data, x, y))
        end

        clusterId = Vector(1:n)
        
        # On associe les données au centroide le plus proche  
        for elem in 1:n

            distances = Vector{Distance}([])

            # On calcule toutes les distances entre cet élement et les centroides
            for centr in centroides
                # if y[elem] == y[centr]
                #     push!(distances, Distance(elem, centr, x))
                # end
                push!(distances, Distance(elem, centr, x))
            end

            # On garde le plus proche centroide
            sort!(distances, by = v -> v.distance)
            distance = distances[1]
            cId1 = clusterId[distance.ids[1]]
            cId2 = clusterId[distance.ids[2]]

            # Si les deux données associées ne sont pas déjà dans le même cluster
            if cId1 != cId2
                

                # Fusionner leurs clusters 
                c1 = clusters[cId1]
                c2 = clusters[cId2]
                merge!(c1, c2)
                for id in c2.dataIds
                    clusterId[id]= cId1
                end

                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end
        end
        #liste des nouveaux centroides
        new_centroides=[]
        
        for id1 in centroides
            cid=clusterId[id1]
            c=clusters[cid]
            xmoy=[0.0 for i in 1:m]

            
            for id in c.dataIds
                for i in 1:m
                    xmoy[i]+= (x[id, i] / length(c.dataIds))
                end

            end
            min=euclidean(x[1, :], xmoy)
            newid=1
            for id2 in 1:n
                if min > euclidean(x[id2, :], xmoy)
                    min = euclidean(x[id2, :], xmoy)
                    newid=id2
                end
            end
            append!(new_centroides,newid)
        end
        sort!(new_centroides)
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end 


function test_kmean(dataSetName; k=3)
    # Préparation des données
    include("../data/" * dataSetName * ".txt") 
    train, test = train_test_indexes(length(Y))
    X_train = X[train,:]
    Y_train = Y[train]
    X_test = X[test,:]
    Y_test = Y[test]

    p = size(X_train, 1)
    clusters = KmeansMerge(X_train, Y_train, k)
    nb_clusters = size(clusters, 1)

    cluster_classes = zeros(Int, p)
    class_nb = 1
    for c in clusters
        dataIds = c.dataIds
        for id in dataIds
            cluster_classes[id] = class_nb
        end
        class_nb = class_nb + 1
    end
    println("$nb_clusters clusters created")
    
    title_name = "k=" * string(k)

    scatter(X_train[:,1], X_train[:,2],
            xlabel = "X",
            ylabel = "Y",
            title = title_name,
            group = cluster_classes,
            legend = false,
            dpi = 1000)
end


function kmeans_julia(x, k)

    n, m = size(x)[1], size(x)[2]

    C = kmeans(transpose(x), k)

    y_algo = assignments(C)

    clusters = Vector{Cluster}([])

    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(y_algo[dataId], x, y_algo))
    end


    for i in 1:k
        for c1 in clusters
            for c2 in clusters
                if c1.class == c2.class
                    merge!(c1, c2)
                end
            end
        end
    end


    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

function test_kmean_julia(dataSetName, k)
    # Préparation des données
    include("../data/" * dataSetName * ".txt") 
    train, test = train_test_indexes(length(Y))
    X_train = X[train,:]
    Y_train = Y[train]
    X_test = X[test,:]
    Y_test = Y[test]

    p = size(X_train, 1)
    clusters = kmeans_julia(X_train, k)
    nb_clusters = size(clusters, 1)

    cluster_classes = zeros(Int, p)
    class_nb = 1
    for c in clusters
        dataIds = c.dataIds
        for id in dataIds
            cluster_classes[id] = class_nb
        end
        class_nb = class_nb + 1
    end
    println("$nb_clusters clusters created")
    
    title_name = "k=" * string(k)

    scatter(X_train[:,1], X_train[:,2],
            xlabel = "X",
            ylabel = "Y",
            title = title_name,
            group = cluster_classes,
            legend = false,
            dpi = 1000)
end


# test_kmean("iris", k=4)
# test_kmean_julia("prnn", k=2)

# test("prnn", minPts = 10, eps = 0.12)
