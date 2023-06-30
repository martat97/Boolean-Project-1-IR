import pandas as pd


# Create a node
class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.child = []


# Tree
class BTree:
    def __init__(self, t):
        self.root = BTreeNode(True)
        self.t = t

    # Insert node
    def insert(self, k, column):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k, column)
        else:
            self.insert_non_full(root, k, column)

    # Insert nonfull
    def insert_non_full(self, x, k, column):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            while i >= 0 and k[column] < x.keys[i][column]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k[column] < x.keys[i][column]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k[column] > x.keys[i][column]:
                    i += 1
            self.insert_non_full(x.child[i], k, column)

    # Split the child
    def split_child(self, x, i):
        t = self.t
        y = x.child[i]
        z = BTreeNode(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t: (2 * t) - 1]
        y.keys = y.keys[0: t - 1]
        if not y.leaf:
            z.child = y.child[t: 2 * t]
            y.child = y.child[0: t - 1]

    # Print the tree
    def print_tree(self, x, l=0):
        print("Level ", l, " ", len(x.keys), end=":")
        for i in x.keys:
            print(i, end=" ")
        print()
        print()
        l += 1
        if len(x.child) > 0:
            for i in x.child:
                self.print_tree(i, l)

    # Scan all the tree to find nodes satisying the condition
    def scan_tree(self, x, column, function1, function2, wildcard, result, l=0):
        child_to_visit = []
        for i in range(0, len(x.keys)):
            if function1(x.keys[i][column], wildcard):
                result = pd.concat([result, pd.DataFrame([x.keys[i]], columns=['term', 'docId', 'rotations'])])
                child_to_visit.append(i)
                if (i == len(x.keys) - 1):
                    child_to_visit.append(i + 1)
            else:
                # greater
                if function2(x.keys[i][column], wildcard):
                    child_to_visit.append(i)
                    break
                else:
                    if (i == len(x.keys) - 1):
                        child_to_visit.append(i + 1)
        if (len(x.child) > 0):
            for i in child_to_visit:
                result = self.scan_tree(x.child[i], column, function1, function2, wildcard, result)

        return result

    def scan_all_tree(self, x, column, function, wildcard, result):
        for row in x.keys:
            if function(row[column], wildcard):
                result = pd.concat([result, pd.DataFrame([row], columns=['term', 'docId', 'rotations'])])

        if len(x.child) > 0:
            for i in x.child:
                result = self.scan_all_tree(i, column, function, wildcard, result)

        return result

    # Search key in the tree
    def search_key(self, k, result, x=None):
        if x is not None:
            i = 0
            while i < len(x.keys) and k > x.keys[i][0]:
                i += 1
            if i < len(x.keys) and k == x.keys[i][0]:
                return pd.concat([result, pd.DataFrame([x.keys[i]], columns=['term', 'docId', 'rotations'])])
            elif x.leaf:
                return result
            else:
                return self.search_key(k, result, x.child[i])

        else:
            return self.search_key(k, result, self.root)


def main():
    # 2*t-1 keys
    # t = ..
    B = BTree(2)
    paroles = pd.DataFrame({"term": ['abac', 'bello', 'belli', 'ciao', 'basso', 'zattera', 'bobo', 'giostra'],
                            "docId": ['1', '2', '3', '4', '5', '6', '7', '8'],
                            "rotations": ['', '', '', '', '', '', '', '']})
    # print(paroles.iloc[0]['term'])

    # parole = ['marta','mary','merenda','mestolo','mesopotamia','mossa','mosse','mulo','mini','morata','sturare', 'zuzzo']

    for i in range(0, len(paroles)):
        B.insert(paroles.iloc[i], 'term')

    B.print_tree(B.root)

    actual_term_phrase = pd.DataFrame(columns=['term', 'docId', 'rotations'])
    key_s = 'basso'
    print(B.search_key(key_s, actual_term_phrase))
    # if B.search_key(key_s) is not None:
    #    print("\nFound")
    # else:
    #    print("\nNot Found")

    # result = pd.DataFrame({"term": [], "docId": [], "rotations":[]})
    # result = B.scan_tree(B.root, 'term', equals, greater,'zu', result)
    # print(result)


if __name__ == '__main__':
    main()
