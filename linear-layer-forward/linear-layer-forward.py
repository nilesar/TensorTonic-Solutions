
def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    ans=[]
    d_in = len(W)
    d_out = len(b)
    for xs in X:
        lis = []
        for i in range(d_out):
            temp = b[i]  
            for k in range(d_in):
                temp += xs[k] * W[k][i]
            lis.append(temp)
        ans.append(lis)
    return ans
            
                