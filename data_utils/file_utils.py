import os
import os.path

def get_ending_of_file(file):
    split_by_pt = file.split(".")
    if len(split_by_pt) == 1:
        raise ValueError(f"there is no ending in file{file}")
    else:
        ending = split_by_pt[-1]
    return ending



def sort_by_prefix(fp):
    file_ls = os.listdir(fp)
    result = {}
    NEW_FRONT = 0

    for file in file_ls:
        print("looping",file)
        front_of_file, ending_of_file = split_by_ending(file)
        category = NEW_FRONT
        smaller_than = []
        larger_than = []
        for front,files in result.items():
            if front.startswith(front_of_file) and front>front_of_file:
                print("appending front",front," to smalelr than ",front_of_file)
                smaller_than.append(front)
            elif front_of_file.startswith(front):
                print("appending front",front," to larger than ",front_of_file)
                larger_than.append(front)
                if len(larger_than)>1:
                    raise ValueError("something is wrong, got ", larger_than, "with ",front,front_of_file)

        if not (smaller_than or larger_than):
           result[front_of_file] = [file]
        elif smaller_than:
            result[front_of_file] = [file]
            for front in smaller_than:
                ls = result.pop(front)
                result[front_of_file].extend(ls)
            print(front_of_file)
        elif larger_than:
            result[larger_than[0]].append(file)

    #sort files
    for files in result.values():
        files.sort()
    return result


def split_by_ending(file_name):
    '''
    like split but adds "" if no ending
    '''
    x = file_name.split(".")
    if len(x)>1:
        return x[-2:]
    else:
        return [x[-1],""]



