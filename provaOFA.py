from NasSearchSpace.ofa.evaluator import OFAEvaluator

ofa = OFAEvaluator(n_classes=10, model_path='NasSearchSpace/ofa/supernets/ofa_resnet50', pretrained=False)
print(ofa.sample())