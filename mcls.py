

if __name__ == '__main__':
    classes = []
    with open('All_ContractTypes.txt', 'r', encoding='utf8') as f:
        while True:
            cls = f.readline()
            if not cls: break
            cls = cls.strip()

            arr = cls.split(' ')
            balpha = True
            for each in arr:
                if not each.isalpha():
                    balpha = False
                    break

            # if cls contains non-alphabeta, don't add
            if not balpha: continue
            # remove duplicates
            if cls in classes: continue

            classes.append(cls)

    classes = sorted(classes, key=lambda cls: len(cls), reverse=False)
    print (classes)

    with open('myclass.txt', 'w') as f:
        for cls in classes:
            f.write(cls + '\n')
