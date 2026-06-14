from sklearn.preprocessing import LabelEncoder

TOKEN_GAP = "-"
TOKEN_GAP_AMP = "X"
TOKENS_AA = list("ARNDCEQGHILKMFPSTWYV")
TOKENS_AHO = sorted([TOKEN_GAP, *TOKENS_AA])
TOKENS_AMP = sorted([TOKEN_GAP_AMP, *TOKENS_AA])

ALPHABET_AHO = LabelEncoder().fit(TOKENS_AHO)
ALPHABET_AMP = LabelEncoder().fit(TOKENS_AMP)

#toy_6 =  [-7,-6,-5, -9, -8,-4,-3,-2,-1,0,1,2,3,4, 8, 9, 5,6,7] #my just previous experiments
toy_6 =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#toy_6 =  LabelEncoder().fit([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9])
#toy_6 =  [7, -5, 9, 1, 2, 4, -8, -7, 3, -2, -1, -3, -4, 6, 0, -9, 5, -6, 8]

#toy_6 =  [-7,-6,-5, -9, -8,-4,-3,-2,-1,0,1,2,3,4, 8, 9, 5,6,7]