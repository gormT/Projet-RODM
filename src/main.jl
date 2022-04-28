include("building_tree.jl")
include("utilities.jl")

function main()

    # Pour chaque jeu de données
    for dataSetName in ["iris", "seeds", "wine", "ecoli", "prnn"]
        
        print("=== Dataset ", dataSetName)

        # Préparation des données
        include("../data/" * dataSetName * ".txt") 
        train, test = train_test_indexes(length(Y))
        X_train = X[train, :]
        Y_train = Y[train]
        X_test = X[test, :]
        Y_test = Y[test]

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes
        time_limit = 30

        # Pour chaque profondeur considérée
        for D in 2:4

            println("  D = ", D)

            ## 1 - Univarié (séparation sur une seule variable à la fois)
            # Création de l'arbre
            print("    Univarié...  \t")
            T, obj, resolution_time, gap = build_tree(X_train, Y_train, D,  multivariate = false, time_limit = time_limit)

            # Test de la performance de l'arbre
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train))
                print("/", prediction_errors(T,X_test,Y_test), "\t")
            end
            println()

            ## 2 - Multivarié
            print("    Multivarié...\t")
            T, obj, resolution_time, gap = build_tree(X_train, Y_train, D, multivariate = true, time_limit = time_limit)
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train))
                print("/", prediction_errors(T,X_test,Y_test), "\t")
            end
            println("\n")
        end
    end 
end
