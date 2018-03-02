with open("annotation.txt", "r") as src:
    with open("annotations_new.txt", "w") as dest:
        master = []
        for line in src:
            lin = line.split(" ")
            if lin[1] == "v":
                v_val = str(lin[2])
                a_val = "0"
            elif lin[1] == "a":
                v_val = "0"
                a_val = str(lin[2])
            outstr = "{0} {1} {2}\n".format(lin[0], v_val, a_val)
            master.append(outstr)
        dest.writelines(master)
