######################################################
################ Code for DynaFlow ###################
######################################################

import sys
import os

data_loc = "batches/batch-primes"
choices_loc = "choices/choices-primes"
suffix = "-defended"

def defend(site, inst, switch_sizes, end_sizes, first_time_gap, poss_time_gaps, subseq_length, memory, suffix_2):
    """Creates defended dataset."""

    # get old sequence
    if inst != None:
        site_data = open("%s/%d-%d" % (data_loc, site, inst), "r")
    else:
        site_data = open("%s/%d" % (data_loc, site), "r")
    old_packets = []
    for line in site_data:
        line = line.split()
        # add [time, direction, length] for each packet
        old_packets.append([float(line[0]), int(line[1]), int(line[3])])

    site_data.close()    

    # make a copy of old packet sequence
    packets = old_packets[:]

    # get new sequence 
    choices = ""
    new_packets = []     
    past_times = []
    past_packets = []
    index = 0
    time_gap = first_time_gap
    # first packet at time zero 
    curr_time = -1 * time_gap

    min_index = 99999999
    while len(packets) != 0 or index not in end_sizes:    

        # get time and direction of next packet
        curr_time = curr_time + time_gap        
        if index % subseq_length == 0:
            curr_dir = 1
        else:
            curr_dir = -1
 
        # add needed packet
        # if possible, packet combination
        packet_size = 0
        num_used_packets = 0
        for i in range(0, len(packets)):
            if packets[i][0] <= curr_time and packets[i][1] == curr_dir and packets[i][2] + packet_size <= 498:
                num_used_packets += 1
                packet_size += packets[i][2]
                if i == 0:
                    past_times.append(packets[i][0])
                past_packets.append(packets[i])
            else:
                break

        del packets[0:num_used_packets]

        new_packets.append([curr_time, curr_dir])

        # find new time gap if time to switch 
        # update string accordingly
        # const for weighted average 
        const = 400
        if index in switch_sizes:
            time_gap_info = find_new_time_gap(past_times, curr_time, time_gap, poss_time_gaps, memory, const)    
            time_gap = time_gap_info[0]
            choices += ("T" + str(time_gap_info[1]))

        # move on to next packet 
        index += 1 

        # get length of defended sequence before any extra padding at end
        if len(packets) == 0 and min_index > index:
            min_index = index 


    # update choices
    choices += ("E" + str(end_sizes.index(index)))    
    choices_stats = open(choices_loc + suffix + suffix_2, "a")
    if inst != None:
        choices_stats.write("%s %s\n" % (site, choices))    
    else:
        choices_stats.write("unmonitored %s\n" % (choices))
    choices_stats.close()

    # write new seq
    new_data_loc = data_loc + suffix + suffix_2
    if not os.path.exists(new_data_loc):
        os.mkdir(new_data_loc)

    if inst != None:
        new_site_data = open("%s/%d-%d" % (new_data_loc, site, inst), "w")
    else:
        new_site_data = open("%s/%d" % (new_data_loc, site), "w")

    for packet in new_packets:
        new_site_data.write("%f %d\n" % (packet[0], packet[1]))
    new_site_data.close()

    return [old_packets[-1][0], new_packets[-1][0], len(old_packets), len(new_packets), min_index]


def find_new_time_gap(past_times, curr_time, time_gap, poss_time_gaps, memory, block_size):
    """Finds new time gap for defended sequence."""

    # find average time gap
    if len(past_times) >= memory:
        average_time_gap = float(past_times[-1] - past_times[-memory]) / (memory - 1)
    elif len(past_times) > 10:
        average_time_gap = float(past_times[-1] - past_times[0]) / (len(past_times) - 1)
    else:
        average_time_gap = time_gap

    # find expected time gap
    exp_packet_num = block_size + 1 * float(curr_time - past_times[-1]) / average_time_gap
    exp_time_gap = block_size / exp_packet_num * average_time_gap

    # choose next timeg gap
    min_diff = 99999
    for i in range(0, len(poss_time_gaps)):
        if min_diff > abs(exp_time_gap - poss_time_gaps[i]):
            min_diff = abs(exp_time_gap - poss_time_gaps[i])
        else:
            return [poss_time_gaps[i - 1], (i - 1)]
    return [poss_time_gaps[-1], len(poss_time_gaps) - 1]


def create_end_sizes(k):
    """Creates list of possible sizes for defended sequence."""   

    end_sizes = []
    for i in range(0, 9999):
        if k ** i > 10000000:
            break
        end_sizes.append(round(k ** i))
    return end_sizes

# closed world
def run_closed(suffix_2, switch_sizes, m):
    """Runs DynaFlow in closed-world."""

    open(choices_loc + suffix + suffix_2, "w").close()

    end_sizes = create_end_sizes(m)

    oldt = 0
    newt = 0
    oldbw = 0
    newbw = 0
    for site in range(0, 100):
        for inst in range(0, 90):
            print site, inst
            ret = defend(site, inst, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
            if ret != None:
                oldt+=ret[0]
                newt+=ret[1]
                oldbw+=ret[2]
                newbw+=ret[3]            


    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print("closed, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    
    results = open("dynaflow.results", "a")  
    results.write("closed, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    results.close()


# open world
def run_open(suffix_2, switch_sizes, m):
    """Runs DynaFlow in open-world."""

    open(choices_loc + suffix + suffix_2, "w").close()

    end_sizes = create_end_sizes(m)

    oldt = 0
    newt = 0
    oldbw = 0
    newbw = 0
    for site in range(0, 100):
        for inst in range(0, 90):
            print site, inst
            ret = defend(site, inst, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
            if ret != None:
                oldt+=ret[0]
                newt+=ret[1]
                oldbw+=ret[2]
                newbw+=ret[3]            


    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print("open-mon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    
    results = open("dynaflow.results", "a")  
    results.write("open-mon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    results.close()


    oldt = 0
    newt = 0
    oldbw = 0
    newbw = 0
    for site in range(0, 9000):
        print site
        ret = defend(site, None, switch_sizes, end_sizes, FIRST_TIME_GAP, POSS_TIME_GAPS, SUBSEQ_LENGTH, MEMORY, suffix_2)
        if ret != None:
            oldt+=ret[0]
            newt+=ret[1]
            oldbw+=ret[2]
            newbw+=ret[3]            

    print newbw
    print oldbw

    toh = float(newt-oldt)/oldt
    bwoh = float(newbw-oldbw)/oldbw

    print("open-unmon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    
    results = open("dynaflow.results", "a")  
    results.write("open-unmon, switch sizes: %s, m: %s, first time gap: %s, poss time gaps: %s, memory: %s\nTOH: %s, BWOH: %s, S: %s\n\n" % (switch_sizes, m, FIRST_TIME_GAP, POSS_TIME_GAPS, MEMORY, toh, bwoh, toh+bwoh))
    results.close()


# tests 

FIRST_TIME_GAP = 0.012
SUBSEQ_LENGTH = 4
MEMORY = 100

POSS_TIME_GAPS = [0.0012, 0.005]
run_closed("-closed-1", [400+100*i for i in range(1000)], 1.002)
run_closed("-closed-2", [400, 1200, 2000, 2800, 3600, 4400, 5200], 1.02)
run_closed("-closed-3", [400, 1200, 2000, 2800, 3600, 4400, 5200], 1.1)
run_closed("-closed-5", [400, 1200, 2000, 2800], 1.1)
run_closed("-closed-6", [400, 1200, 2000, 2800], 1.2)
POSS_TIME_GAPS = [0.0015]
run_closed("-closed-4", [400], 1.2)
run_closed("-closed-7", [400], 1.02)
run_closed("-closed-8", [400], 1.1)
run_closed("-closed-9", [400], 1.5)
run_closed("-closed-10", [400], 2)
run_closed("-closed-11", [400], 2.3)

POSS_TIME_GAPS = [0.0012, 0.005]
run_open("-open-1", [400+100*i for i in range(1000)], 1.002)
run_open("-open-2", [400, 1200, 2000, 2800, 3600, 4400, 5200], 1.02)
run_open("-open-3", [400, 1200, 2000, 2800, 3600, 4400, 5200], 1.1)
run_open("-open-5", [400, 1200, 2000, 2800], 1.1)
run_open("-open-6", [400, 1200, 2000, 2800], 1.2)
POSS_TIME_GAPS = [0.0015]
run_open("-open-4", [400], 1.2)
run_open("-open-7", [400], 1.02)
run_open("-open-8", [400], 1.1)
run_open("-open-9", [400], 1.5)
run_open("-open-10", [400], 2)
run_open("-open-11", [400], 2.3)



