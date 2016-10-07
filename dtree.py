import math
from collections import Counter, OrderedDict
from random import shuffle, seed
from mldata import *

'''
Written by Nick Stevens
9/24/2016
'''

# Useful constants
INFINITY = float('inf')
NEG_INFINITY = -float('inf')
CLASS_LABEL = -1


class DTree(object):

    def __init__(self, example_set, schema, max_depth, ig_option):
        self.full_example_set = example_set
        self.MIN_EXAMPLES = math.frexp(len(example_set))[1]-1  # log base 2 of the number of examples
        self.full_schema = schema
        self.full_feature_set = [f for f in schema[:CLASS_LABEL] if f.type != 'ID']
        self.class_label = self.full_schema[CLASS_LABEL]
        self.class_label_index = len(self.full_schema)-1
        self.max_depth = max_depth
        self.ig_option = ig_option

    def build_tree(self, example_set=None, feature_set=None, depth=0, size=0, parent_default=None):
        # Default values so that this method can be easily called from outside class
        if example_set is None:
            example_set = self.full_example_set
        if feature_set is None:
            feature_set = self.full_feature_set

        num_examples = len(example_set)

        if num_examples == 0:
            # Used to print tree structure
            #print(' |   '*depth + str(parent_default) + ' (' + str(num_examples) + ' examples)')
            return DTreeNode(None, parent_default), depth, size+1

        # If there are examples, calculate the default class label at this level
        default_label = self.most_common_feature_value(example_set, self.class_label)

        # Check if all examples have the same class label
        label_dist = self.label_value_distribution(example_set)
        for label in label_dist.keys():
            if label_dist[label] == num_examples:
                # Used to print tree structure
                #print(' |   '*depth + str(label) + ' (' + str(num_examples) + ' examples)')
                return DTreeNode(None, label), depth, size+1

        # Check if the schema is empty or if there are too few examples to split on
        if self.depth_limit_reached(depth) or len(feature_set) == 0 or num_examples < self.MIN_EXAMPLES:
            # Used to print tree structure
            #print(' |   '*depth + str(default_label) + ' (' + str(num_examples) + ' examples)')
            return DTreeNode(None, default_label), depth, size+1

        # Pick the current best feature for this node and recurse
        else:
            best_feature, feature_dist, gain = self.max_gain(example_set, num_examples, feature_set)
            # Used to print tree structure
            #print(' |   '*depth + best_feature.name + ' (' + str(num_examples) + ' examples, gain='+ str("%.4f" % gain) + ')')
            node = DTreeNode(best_feature, None)
            feature_subset = list(feature_set)
            lower_bound = NEG_INFINITY
            max_depth = depth + 1
            if best_feature.type == 'CONTINUOUS':
                feature_values = feature_dist.keys()
            else:
                # If best feature is non-continuous, remove from feature list for future branches
                feature_subset.remove(best_feature)
                feature_values = best_feature.values
            for value in feature_values:
                example_subset = self.feature_value_subset(example_set, best_feature, value, lower_bound)
                subtree_root, subtree_depth, size = self.build_tree(example_subset, feature_subset,
                                                                    depth+1, size, default_label)
                if subtree_depth > max_depth:
                    max_depth = subtree_depth
                node.add_child(value, subtree_root)
                size += 1
                lower_bound = value
            return node, max_depth, size

    def depth_limit_reached(self, depth):
        return 0 < self.max_depth <= depth

    def most_common_feature_value(self, example_set, feature):
        feature_dist = self.feature_value_distribution(example_set, feature)
        return max(feature_dist, key=lambda key: feature_dist.get)

    # Feature with the highest Information Gain or Gain Ratio
    def max_gain(self, example_set, example_set_length, feature_set):
        max_feature = feature_set[0]
        max_gain = 0.0
        class_entropy = self.entropy(example_set, example_set_length)
        max_dist = {}
        for feature in feature_set:
            dist = self.feature_value_distribution(example_set, feature)
            gain = self.gain(example_set, example_set_length, feature, dist, class_entropy)
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
                max_dist = dist
        return max_feature, max_dist, max_gain

    # Single method for info gain and gain ratio. Calculates intrinsic_value
    # and info_gain in the same loop to make gain_ratio calculation quicker
    def gain(self, example_set, example_set_length, feature, feature_dist, class_entropy):
        if not feature_dist or example_set_length < self.MIN_EXAMPLES:
            return 0.0
        num_examples = float(example_set_length)
        info_gain = class_entropy
        intrinsic_value = 0.0
        lower_bound = NEG_INFINITY
        for value in feature_dist.keys():
            example_subset = self.feature_value_subset(example_set, feature, value, lower_bound)
            example_subset_length = len(example_subset)
            try:
                value_prob = example_subset_length / num_examples
            except ZeroDivisionError:
                break
            value_entropy = self.entropy(example_subset, example_subset_length)
            info_gain -= value_prob * value_entropy
            try:
                intrinsic_value -= value_prob * math.log(value_prob, 2)
            except ValueError:
                pass  # Reached if value_prob too close to 0
            lower_bound = value
        if self.ig_option == 1 and intrinsic_value != 0.0:
            gain = info_gain / intrinsic_value
        else:
            gain = info_gain
        return gain

    def entropy(self, example_set, example_set_length, label_dist=None):
        num_examples = float(example_set_length)
        entropy = 0.0
        if label_dist is None:
            label_dist = self.label_value_distribution(example_set)
        for count in label_dist.values():
            try:
                value_prob = count/num_examples
            except ZeroDivisionError:
                return entropy
            try:
                entropy -= value_prob * math.log(value_prob, 2)
            except ValueError:
                pass  # Reached if value_prob too close to 0
        return entropy

    def feature_value_subset(self, example_set, feature, value, lower_bound=NEG_INFINITY):
        i = self.feature_index(feature)
        if feature.type == 'CONTINUOUS':
            # For continuous distributions, value is the upper bound
            return [ex for ex in example_set if lower_bound < ex[i] <= value]
        else:
            return [ex for ex in example_set if ex[i] == value]

    def feature_value_distribution(self, example_set, feature):
        i = self.feature_index(feature)
        if feature.type == 'CONTINUOUS':
            return self.best_split_dist(example_set, i)
        else:
            return self.discrete_value_distribution(example_set, i)

    def label_value_distribution(self, example_set):
        if self.class_label.type != 'CONTINUOUS':
            return self.discrete_value_distribution(example_set, self.class_label_index)
        else:
            return {}  # Code does not handle continuous class labels

    def discrete_value_distribution(self, example_set, index):
        dist = {}
        get = dist.get
        for example in example_set:
            try:
                value = example[index]
                dist[value] = get(value, 0) + 1
            except ValueError:
                pass
        return dist

    def best_split_dist(self, example_set, index):
        # Sort example_set on feature at index
        sorted_examples = sorted(example_set, key=lambda x: x[index])
        num_examples = len(sorted_examples)
        class_entropy = self.entropy(sorted_examples, num_examples)
        max_gain = 0.0
        max_dist = OrderedDict()
        if num_examples < self.MIN_EXAMPLES:
            return max_dist
        split_indices = self.get_split_indices(sorted_examples, num_examples, index)
        num_splits = len(split_indices)
        split_candidates = self.get_split_candidates(split_indices, num_splits)
        prev_num_below = split_candidates[split_candidates.keys()[0]]
        prev_below_dist = None
        prev_above_dist = None
        for split in split_candidates:
            dist = OrderedDict()
            num_below = split_candidates[split]
            dist[split] = num_below
            dist[INFINITY] = num_examples - num_below
            diff = num_below - prev_num_below
            gain, prev_below_dist, prev_above_dist = \
                self.split_gain(sorted_examples, num_examples, dist[split], dist[INFINITY],
                                diff, prev_below_dist, prev_above_dist, class_entropy)
            if gain > max_gain:
                max_gain = gain
                max_dist = dist
            prev_num_below = num_below
        if not max_dist:
            max_dist[NEG_INFINITY] = 0
            max_dist[INFINITY] = num_examples
        return max_dist

    # Returns all indices where adjacent examples sorted by feature value have differing class labels
    def get_split_indices(self, sorted_examples, example_set_length, index):
        split_indices = OrderedDict()
        for i in xrange(0, example_set_length-1):
            curr_ex = sorted_examples[i]
            next_ex = sorted_examples[i+1]
            if curr_ex[CLASS_LABEL] != next_ex[CLASS_LABEL]:
                split_indices[(curr_ex[index]+next_ex[index])/2.0] = i+1
        return split_indices

    # Returns a logarithmically-scaled-down subset of the split indices
    def get_split_candidates(self, split_indices, num_splits):
        threshold = self.MIN_EXAMPLES
        # Step size is log base 2 of split-list length for sufficiently large lists
        step = (1 if num_splits <= threshold else math.frexp(num_splits)[1]-1)
        split_candidates = OrderedDict()
        split_keys = split_indices.keys()
        for x in xrange(0, num_splits, step):
            split = split_keys[x]
            split_candidates[split] = split_indices[split]
        return split_candidates

    # Lightweight gain computation for binary splits of an example set.
    # Only computes the class label distribution for the small section of examples
    # between this split and the previous split. This is used to update the distributions
    # from the previous split to reflect the distributions in this split. This cuts out
    # most of the expensive calls to discrete_value_distribution.
    def split_gain(self, example_set, example_set_length, num_below, num_above,
                   diff_from_prev, prev_below_dist, prev_above_dist, class_entropy):
        assert example_set_length != 0
        num_examples = float(example_set_length)
        info_gain = class_entropy
        below_subset = example_set[:num_below]
        above_subset = example_set[-num_above:]
        if diff_from_prev > 0:
            below_dist = prev_below_dist
            above_dist = prev_above_dist
            diff_dist = self.label_value_distribution(below_subset[-diff_from_prev:])
            below_get = below_dist.get
            above_get = above_dist.get
            for value in diff_dist.keys():
                try:
                    diff_count = diff_dist[value]
                    below_dist[value] = below_get(value, 0) + diff_count
                    above_dist[value] = above_get(value, diff_count) - diff_count
                except ValueError:
                    pass
        else:
            below_dist = self.label_value_distribution(below_subset)
            above_dist = self.label_value_distribution(above_subset)
        # Subtract (probability * entropy) for subset below split value
        below_prob = num_below/num_examples
        below_entropy = self.entropy(below_subset, num_below, below_dist)
        info_gain -= below_prob * below_entropy
        # Subtract (probability * entropy) for subset above split value
        above_prob = num_above/num_examples
        above_entropy = self.entropy(above_subset, num_above, above_dist)
        info_gain -= above_prob * above_entropy
        return info_gain, below_dist, above_dist

    def feature_index(self, feature):
        try:
            return self.full_schema.index(feature)
        except ValueError:
            return -1


class DTreeNode(object):
    # Internal nodes are described by feature, leaf nodes by label
    def __init__(self, feature=None, label=None):
        assert not(feature is None and label is None)
        self.feature = feature
        self.label = label
        if self.feature is not None and self.feature.type == 'CONTINUOUS':
            self.children = OrderedDict()
        elif self.feature is not None:
            self.children = {}
        else:
            self.children = None  # label nodes are leaf nodes

    # Adds a child node associated with a value of this DTreeNode's feature
    def add_child(self, value, node):
        assert self.children is not None
        self.children[value] = node


def main():
    options = sys.argv[1:]

    assert len(options) >= 1
    file_base = options[0]
    example_set = parse_c45(file_base)
    schema = example_set.schema

    # Default values in case some arguments not given
    cv_option = 0  # If 0, use cross-validation. If 1, run algorithm on full sample.
    max_depth = 0  # Max depth of tree. Nonnegative integer. If zero, grow full tree.
    ig_option = 0  # If 0, use information gain as split critereon. If 1, use gain ratio.

    if len(options) >= 2:
        cv_option = (1 if options[1] == '1' else 0)
    if len(options) >= 3:
        max_depth = (int(options[2]) if int(options[2]) > 0 else 0)
    if len(options) >= 4:
        ig_option = (1 if options[3] == '1' else 0)

    if cv_option == 0:
        # Cross-validation
        fold_set = k_folds_stratified(example_set, 5)
        accuracy = run_cross_validation(fold_set, schema, max_depth, ig_option)
        print('Average accuracy: ' + str(accuracy))
    else:
        # Run algorithm on full sample
        print('Building decision tree\n')
        d_tree = DTree(example_set, schema, max_depth, ig_option)
        root, depth, size = d_tree.build_tree()
        print('Measuring tree accuracy\n')
        accuracy = measure_tree_accuracy(example_set, schema, root)
        print(root.feature.name)
        print(root.feature.type)
        print('Depth: ' + str(depth))
        print('Size: ' + str(size))
        print('Accuracy: ' + str(accuracy) + '\n')


def k_folds_stratified(example_set, k):
    seed(12345)
    shuffle(example_set)
    label_dist = Counter(ex[CLASS_LABEL] for ex in example_set)
    label_values = label_dist.keys()
    examples_with_label = [[] for x in xrange(len(label_values))]
    # Get the set of examples for each label
    for example in example_set:
        for label in label_values:
            if example[CLASS_LABEL] == label_values[label]:
                examples_with_label[label].append(example)
                break
    # Group examples by class label
    sorted_examples = []
    for example_subset in examples_with_label:
        sorted_examples += example_subset
    folds = [[] for x in xrange(k)]
    # Distribute sorted examples evenly amongst all k folds
    for i in xrange(0, len(sorted_examples)):
        assigned_fold = i % k
        folds[assigned_fold].append(sorted_examples[i])
    for fold in folds:
        shuffle(fold)
    return folds


def run_cross_validation(fold_set, schema, max_depth, ig_option):
    k = len(fold_set)
    average_accuracy = 0.0
    for i in xrange(0, k):
        validation_set = fold_set[i]
        training_set = []
        for j in xrange(1, k):
            training_set += fold_set[(i+j) % k]
        print('Building decision tree\n')
        d_tree = DTree(training_set, schema, max_depth, ig_option)
        root, depth, size = d_tree.build_tree()
        print('Measuring tree accuracy\n')
        accuracy = measure_tree_accuracy(validation_set, schema, root)
        print('Fold ' + str(i+1))
        print(root.feature.name)
        print(root.feature.type)
        print('Depth: ' + str(depth))
        print('Size: ' + str(size))
        print('Accuracy: ' + str(accuracy) + '\n')
        average_accuracy += (1/float(k)) * accuracy
    return average_accuracy


def measure_tree_accuracy(example_set, schema, root_node):
    num_examples = float(len(example_set))
    misclassified = 0
    for example in example_set:
        actual_label = example[CLASS_LABEL]
        assigned_label = classify_example(example, schema, root_node)
        if assigned_label != actual_label:
            misclassified += 1
    accuracy = 1 - (misclassified / num_examples)
    return accuracy


def classify_example(example, schema, node):
    if node.label is not None:
        return node.label
    elif node.feature is not None:
        i = schema.index(node.feature)
        if node.feature.type != 'CONTINUOUS':
            next_node = node.children[example[i]]
        else:
            keys = node.children.keys()
            next_node = node.children[keys[0]] if example[i] <= keys[0] else node.children[keys[1]]
        return classify_example(example, schema, next_node)
    return None


if __name__ == "__main__":
    main()
