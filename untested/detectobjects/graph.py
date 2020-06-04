class graph:
    def __init__(self):
        self.N = set()
        self.E = set()

    def from_matrix(self, A):
        Y, X = A.shape

        for y in range(Y-1):
            for x in range(X-1):
                self.N.add(A[y,x])

                if A[y,x] != A[y+1,x]:
                    self.E.add((A[y,x], A[y+1,x], 1))
                    self.E.add((A[y+1,x], A[y,x], 1))
                if A[y,x] != A[y, x+1]:
                    self.E.add((A[y,x], A[y,x+1], 1))
                    self.E.add((A[y,x+1], A[y,x], 1))
