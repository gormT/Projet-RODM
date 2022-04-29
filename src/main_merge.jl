include("building_tree.jl")
include("utilities.jl")
include("merge.jl")

function main_merge()
    for dataSetName in ["iris", "seeds", "wine", "ecoli", "prnn"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include("../data/" * dataSetName * ".txt") 
        train, test = train_test_indexes(length(Y))
        X_train = X[train,:]
        Y_train = Y[train]
        X_test = X[test,:]
        Y_test = Y[test]

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 10

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            testMerge2(X_train, Y_train, X_test, Y_test, D, time_limit = time_limit, isMultivariate = false)
            println("\t\tMultivarié")
            testMerge2(X_train, Y_train, X_test, Y_test, D, time_limit = time_limit, isMultivariate = true)
        end
    end
end 

function testMerge(X_train, Y_train, X_test, Y_test, D; time_limit::Int=-1, isMultivariate::Bool = false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for gamma in 0:0.2:1
        print("\t\t\t", gamma * 100, "%\t\t")
        clusters = simpleMerge(X_train, Y_train, gamma)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train))
        print("/", prediction_errors(T,X_test,Y_test), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end 

function testMerge2(X_train, Y_train, X_test, Y_test, D; time_limit::Int=-1, isMultivariate::Bool = false)

    minPts = 4  # nombre de voisins minimum pour qu'une région soit considérée comme "dense"
    # Pour tout pourcentage de regroupement considéré
    println("\t\t\teps\t\t# clusters\tGap")
    for eps in 0:0.2:0.8
        print("\t\t\t", eps, "\t\t")
        clusters = dbscanMerge(X_train, Y_train, minPts, eps)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train))
        print("/", prediction_errors(T,X_test,Y_test), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end 

