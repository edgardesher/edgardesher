import itertools


class RuleMiner(object):

    def __init__(self, support_t, confidence_t, relative_support=False):
        """
        Class constructor for RuleMiner.
        Arguments:
            support_t {float or int} -- support threshold for the dataset.
                - If relative_support is True, this should be a float (e.g., 0.1 for 10%).
                - If relative_support is False, this should be an int (e.g., 100 transactions).
            confidence_t {float} -- confidence threshold for the dataset (0 to 1).
            relative_support {bool} -- whether the support threshold is relative (default: False).
        """
        self.support_t = support_t #Support threshold value for the itemsets
        self.confidence_t = confidence_t #Confidence threshold for the rule mining
        self.relative_support = relative_support #Whether the support is relative or absolute

    def get_support(self, syn_df, itemset):
        """
        Returns the support for an itemset as a count or proportion.
        Arguments:
            syn_df {pd.DataFrame} -- DataFrame containing the dataset.
            itemset {list} -- list of items to check in each observation.
        Returns:
            float -- relative support (if relative_support is True).
            int -- absolute support (if relative_support is False).
        """
        try:
            #Select rows where all values in the itemset are True
            itemset_present = syn_df[itemset].all(axis=1) #Checks if all items in itemset are present in the row
            absolute_support = itemset_present.sum() #Count how many times the itemset appears in the DataFrame

            #Calculate relative support if required
            if self.relative_support:
                total_transactions = len(syn_df) #Get the total number of transactions (rows)
                return absolute_support / total_transactions  #Proportion of total dataset that contains itemset
            else:
                return absolute_support  #Return the absolute count of itemset occurrences

        except KeyError as e:
            print(f"Error: Column names {e} not found in syn_df.") #Error handling if columns not found
            return 0

    def merge_itemsets(self, itemsets):
        """Returns a list of merged itemsets. If one itemset of size 2
        from itemsets contains one item in another itemset of size 2 from
        itemsets, the function merges these itemsets to produce an itemset of
        size 3.
        Arguments:
            itemsets {list} -- list which contains itemsets to merge.
        Returns:
            list -- list of merged itemsets

        Example:
            If itemsets is equal to [[1, 2], [1, 3], [1, 5], [2, 6]], then the
            function should return [[1, 2, 3], [1, 2, 5], [1, 2, 6], [1, 3, 5]]
        """

        new_itemsets = [] #Initialize the list for storing merged itemsets

        cur_num_items = len(itemsets[0]) #Get the number of items in the current itemsets

        if cur_num_items == 1:
            for i in range(len(itemsets)): #Iterate through each itemset
                for j in range(i + 1, len(itemsets)): #Compare with subsequent itemsets to merge
                    new_itemsets.append(list(set(itemsets[i]) | set(itemsets[j]))) #Merge by union of sets

        else:
            for i in range(len(itemsets)): #Loop through all itemsets
                for j in range(i + 1, len(itemsets)): #Compare each itemset with the subsequent ones
                    combined_list = list(set(itemsets[i]) | set(itemsets[j])) #Merge two itemsets into a combined itemset
                    combined_list.sort() #Sort the combined itemset to maintain order
                    if len(combined_list) == cur_num_items + 1 and combined_list not in new_itemsets:
                        new_itemsets.append(combined_list) #Append the merged itemset to the new list if it's unique

        return new_itemsets #Return the list of merged itemsets

    def get_rules(self, itemset):
        """Returns a list of rules produced from an itemset.
        Arguments:
            itemset {list} -- list which contains items.
        Returns:
            list -- list of rules produced from an itemset.
        """
        rules = [] #Initialize an empty list to store the rules
        
        # Generate all non-empty subsets of the itemset
        for r in range(1, len(itemset)): #Loop through all possible sizes for the left-hand side (LHS) subsets
            for lhs in itertools.combinations(itemset, r):  #Left-hand side (LHS) is a combination
                lhs = list(lhs) #Convert combination into a list
                rhs = list(set(itemset) - set(lhs))  #Right-hand side (RHS) is the remaining items
                rules.append([lhs, rhs])  #Append the rule to the list
        return rules #Return the list of generated rules

    def get_frequent_itemsets(self, data):
        """Returns a list of frequent itemsets in the dataset. The support of each
        frequent itemset should be greater than or equal to the support threshold.
        Arguments:
            data {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
        Returns:
            list -- list of frequent itemsets in the dataset.
        """

        #Start with single-item itemsets
        itemsets = [[i] for i in data.columns]
        old_itemsets = []
        flag = True

        while flag:
            new_itemsets = []
            
            #Check each itemset for support
            for itemset in itemsets:
                #Get support for the itemset
                support = self.get_support(data, itemset)
                
                #If support meets or exceeds threshold, add to new_itemsets
                if support >= self.support_t:
                    new_itemsets.append(itemset)

            #If we found new frequent itemsets, continue expanding
            if new_itemsets:
                old_itemsets = new_itemsets
                itemsets = self.merge_itemsets(new_itemsets)
            else:
                #No more frequent itemsets can be generated
                flag = False
                itemsets = old_itemsets

        return itemsets

    def get_confidence(self, syn_df, rule):
        """Returns the confidence for a rule. Suppose the rule is X -> Y,
        then the confidence for the rule is the support of the concatenated
        list of X and Y divided by the support of X.
        Arguments:
            syn_df {pd.DataFrame} -- DataFrame containing the dataset
            rule {list} -- Rule in the format [lhs, rhs]
        Returns:
            float -- confidence for rule in syn_df
        """
        #Unpack lhs and rhs from the rule
        lhs, rhs = rule
        
        #Concatenate lhs and rhs for the full itemset
        full_itemset = lhs + rhs

        #Calculate supports
        support_full = self.get_support(syn_df, full_itemset)
        support_lhs = self.get_support(syn_df, lhs)

        #Calculate and return confidence
        if support_lhs == 0:
            return 0
        return support_full / support_lhs

    def get_association_rules(self, syn_df):
        """Returns a list of association rules with support greater than or
        equal to the support threshold support_t and confidence greater than or
        equal to the confidence threshold confidence_t.
        Arguments:
            syn_df {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
        Returns:
            list -- list of association rules. If the rule is X -> y, then each
            rule is a list containing [X, y].
        """
        #Get frequent itemsets from the dataset
        itemsets = self.get_frequent_itemsets(syn_df)

        #List to store all generated rules
        all_rules = []

        #Generate rules for each itemset
        for itemset in itemsets:
            # Get all rules from the current itemset
            itemset_rules = self.get_rules(itemset)
            all_rules.extend(itemset_rules)

        #List to store rules that meet the support and confidence thresholds
        valid_rules = []

        #Filter rules by support and confidence thresholds
        for rule in all_rules:
            lhs, rhs = rule
            #Get support for the rule (i.e., lhs + rhs)
            support = self.get_support(syn_df, lhs + rhs)
            
            #Check if the support meets the threshold
            if support >= self.support_t:
                #Get the confidence of the rule (i.e., support(lhs + rhs) / support(lhs))
                lhs_support = self.get_support(syn_df, lhs)
                if lhs_support > 0:  #Avoid division by zero
                    confidence = support / lhs_support
                    if confidence >= self.confidence_t:
                        valid_rules.append(rule)

        return valid_rules