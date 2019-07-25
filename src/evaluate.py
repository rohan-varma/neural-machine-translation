from collections import defaultdict
def build_reference_dict(sentences):
	# sentences is a list of lists, where each list is a translation
	# li[0] is english, li[1] is french, but there may be multiple translations of the same english phrase
	# so return a dict of key: english, value: list of french translations (reference translations)
	ref_dict = defaultdict(list)
	for translation in sentences:
		english, french = translation
		ref_dict[english].append(french)
	return ref_dict