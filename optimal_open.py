# Optimal attacker TPR, FPR, precision, F1 score -- open-world

import math

def get_stats(prop, choices_stats_loc):
    """Computes TPR, FPR, precision, F1-score of optimal attacker (in open-world)."""

    choices_stats = open(choices_stats_loc, "r")
    
    # get list of sites and choices
    site_list = []
    trace_list = []
    num_inst_list = []
    num_inst = 0
    last_site = None
    for line in choices_stats:
        line = line.split()
        if line[0] not in site_list:
            site_list.append(line[0])
        if line[1] not in trace_list:
            trace_list.append(line[1])
        if line[0] == last_site:
            num_inst += 1 
        else:
            if num_inst != 0:
                num_inst_list.append(num_inst)
            num_inst = 1
        last_site = line[0]
    num_inst_list.append(num_inst)

    # compute probability of visiting each site
    site_prob_list = [float(prop) / (len(site_list) - 1) for i in range(0, len(site_list) - 1)]
    site_prob_list.append(1.0 - float(prop))
    
    # compute probability of visiting each site, trace pair 
    prob_array = [[0 for j in range(0, len(trace_list))] for i in range(0, len(site_list))]
    choices_stats.seek(0)
    for line in choices_stats:
        line = line.split()
        i = site_list.index(line[0])
        j = trace_list.index(line[1])
        prob_array[i][j] += (1.0 / num_inst_list[i] * site_prob_list[i])
    
    # compute guess given trace
    guess_list = []
    for j in range(0, len(trace_list)):
        max_prob = -1
        guess = -1
        for i in range(0, len(site_list)):
            if max_prob < prob_array[i][j]:
                max_prob = prob_array[i][j]
                guess = i
        guess_list.append(guess)
    
    # tpr
    tpr_num = 0
    tpr_den = 0
    for i in range(0, len(site_list) - 1):
        for j in range(0, len(trace_list)):
            if guess_list[j] == i:
                phi = 1
            else:
                phi = 0
    
            tpr_num += prob_array[i][j] * phi
            tpr_den += prob_array[i][j]
    
    tpr = tpr_num / tpr_den
    
    # fpr, prec, f1
    fpr_num = 0
    fpr_den = 0
    for j in range(0, len(trace_list)):
        if guess_list[j] == len(site_list) - 1:
            phi = 1
        else:
            phi = 0
    
        fpr_num += prob_array[-1][j] * (1.0 - phi)
        fpr_den += prob_array[-1][j]

    if fpr_den == 0:
        fpr = "N/A, closed-world"    
    else:
        fpr = fpr_num / fpr_den

    try:
        prec = (tpr*prop) / float(tpr*prop + fpr*(1-prop))
    except:
        prec = 0

    try:
        f1 = (2*prec*tpr)/float(prec+tpr)
    except:
        f1 = 0

    print "Loc:", choices_stats_loc    
    print "TPR:", round(tpr,3)
    print "FPR:", round(fpr,3)
    print "Precision:", round(prec,3)
    print "F1:", round(f1,3)
    print 


# sample tests
for i in range(1, 12):
    get_stats(0.5, "choices/choices-primes-defended-open-%s" % (i))


