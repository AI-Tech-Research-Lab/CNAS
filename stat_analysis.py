from scipy.stats import friedmanchisquare

# Open log file for writing
with open("friedman_results.log", "w") as log:

    # Test 1: MACs on SVHN
    NACHOS   = [1.49, 1.42, 1.5, 1.47, 1.44]
    NACHOS_NP = [1.44, 1.48, 1.42, 1.46, 1.45]
    NACHOS_NC = [1.84, 1.9, 2, 1.92, 2.1]
    EDANAS    = [1.48, 1.44, 1.47, 1.48, 1.46]

    stat1, p1 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)

    log.write("🔬 Friedman Test 1: Average MACs on SVHN\n")
    log.write(f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat1:.4f}\n")
    log.write(f"P-value: {p1:.4f}\n")
    if p1 < 0.05:
        log.write("Result: Significant differences detected among methods.\n\n")
    else:
        log.write("Result: No significant difference among methods.\n\n")

    # Test 2: Accuracy on SVHN
    NACHOS   = [79.98, 78.9, 80.1, 79.6, 81.2]
    NACHOS_NP = [77.18, 76.7, 76.0, 77.7, 77.12]
    NACHOS_NC = [84.87, 84.0, 84.3, 84.6, 85.2]
    EDANAS= [78.1,77.6,77.9,78.2,78.1]

    stat2, p2 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)

    log.write("🎯 Friedman Test 2: Accuracy on SVHN\n")
    log.write(f"Methods: NACHOS, NACHOS_NP, NACHOS_NC EDANAS\n")
    log.write(f"Chi-squared statistic: {stat2:.4f}\n")
    log.write(f"P-value: {p2:.4f}\n")
    if p2 < 0.05:
        log.write("Result: Significant differences detected among methods.\n")
    else:
        log.write("Result: No significant difference among methods.\n")
    
    log.write("\n")

    # Test 3: MACs on CIFAR-10
    NACHOS   = [2.44,2.43,2.46,2.44,2.4]
    NACHOS_NP = [2.36,2.44,2.45,2.4,2.33]
    NACHOS_NC = [2.81, 2.9, 3.1, 2.9, 2.84]
    EDANAS    = [2.46, 2.48, 2.47, 2.45, 2.48]

    stat3, p3 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🔬 Friedman Test 3: Average MACs on CIFAR-10\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat3:.4f}\n")
    log.write(f"P-value: {p3:.4f}\n")
    if p3 < 0.05:
        log.write("Result: Significant differences detected among methods.\n\n")
    else:
        log.write("Result: No significant difference among methods.\n\n")
    
    # Test 4: Accuracy on CIFAR-10
    NACHOS   = [72.86, 71.9, 72.5, 73.2, 72.8]
    NACHOS_NP = [71.72, 71.4, 70.6, 71.8, 71.4]
    NACHOS_NC = [72.18, 72.3, 72.6, 72.05, 72.12]
    EDANAS = [67.76, 67.68, 67.99, 67.72, 67.77]

    stat4, p4 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🎯 Friedman Test 4: Accuracy on CIFAR-10\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat4:.4f}\n")
    log.write(f"P-value: {p4:.4f}\n")
    if p4 < 0.05:
        log.write("Result: Significant differences detected among methods.\n")
    else:
        log.write("Result: No significant difference among methods.\n")
    log.write("\n")

    # Test 5: MACs on CINIC-10
    NACHOS = [2.42, 2.42, 2.43, 2.4, 2.41]
    NACHOS_NP = [2.33, 2.42, 2.37, 2.45, 2.46]
    NACHOS_NC = [3.82, 3.6, 3.9, 3.78, 3.62]
    EDANAS = [2.43, 2.4, 2.48, 2.47, 2.46]

    stat5, p5 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🔬 Friedman Test 5: Average MACs on CINIC-10\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat5:.4f}\n")
    log.write(f"P-value: {p5:.4f}\n")
    if p5 < 0.05:
        log.write("Result: Significant differences detected among methods.\n\n")
    else:
        log.write("Result: No significant difference among methods.\n\n")
    
    # Test 6: Accuracy on CINIC-10
    NACHOS = [60.77, 60.4, 61.2, 60.9, 61]
    NACHOS_NP = [60.44, 59.6, 60.38, 60.1, 60.3]
    NACHOS_NC = [59.86, 59.98, 60.1, 60.05, 59.82]
    EDANAS = [60.52, 60.56, 60.55, 60.9, 60.7]

    stat6, p6 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🎯 Friedman Test 6: Accuracy on CINIC-10\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat6:.4f}\n")
    log.write(f"P-value: {p6:.4f}\n")
    if p6 < 0.05:
        log.write("Result: Significant differences detected among methods.\n")
    else:
        log.write("Result: No significant difference among methods.\n")
    log.write("\n")

    # Test 7: MACs on Imagenette

    NACHOS = [92.74, 92.68, 91.4, 92.74, 92.12]
    NACHOS_NP = [84.2, 84.6, 84.5, 83.9, 84.7]
    NACHOS_NC = [118.71, 116.23, 117.56, 120.11, 117.43]
    EDANAS = [81.03, 83.46, 80.67, 82.4, 82.02]

    stat7, p7 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🔬 Friedman Test 7: Average MACs on Imagenette\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat7:.4f}\n")
    log.write(f"P-value: {p7:.4f}\n")
    if p7 < 0.05:
        log.write("Result: Significant differences detected among methods.\n\n")
    else:
        log.write("Result: No significant difference among methods.\n\n")

    # Test 8: Accuracy on Imagenette
    NACHOS = [94.93, 95.1, 94.8, 94.9, 94.7]
    NACHOS_NP = [85.41, 85.2, 85.1, 85.11, 85.4]
    NACHOS_NC = [97.36, 97.12, 97.24, 97.4, 97.33]
    EDANAS = [91.12, 91.37, 90.89, 90.3, 90.1]

    stat8, p8 = friedmanchisquare(NACHOS, NACHOS_NP, NACHOS_NC, EDANAS)
    log.write("🎯 Friedman Test 8: Accuracy on Imagenette\n"
              f"Methods: NACHOS, NACHOS_NP, NACHOS_NC, EDANAS\n")
    log.write(f"Chi-squared statistic: {stat8:.4f}\n")
    log.write(f"P-value: {p8:.4f}\n")
    if p8 < 0.05:
        log.write("Result: Significant differences detected among methods.\n")
    else:
        log.write("Result: No significant difference among methods.\n")
    log.write("\n")

print("Friedman test results saved to 'friedman_results.log'")


