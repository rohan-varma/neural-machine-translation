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

def modified_n_gram_precision(candidate, references):
	# candidate is a list giving the candidate translation,
	# references is a list of lists with the actual translations.
	# first, for each word, compute max_ref_count and store it in a dict.
	max_ref_counts = {}
	for word in candidate:
		# no need to recount if we've already done the counting for the same word previously
		if word in max_ref_counts:
			continue
		# compute the max number of times it occurs in references
		max_occs = 0
		for ref in references:
			# compute number of times it occurs
			count = sum([1 if w == word else 0 for w in ref])
			max_occs = max(max_occs, count)
		max_ref_counts[word] = max_occs
	# count the number of candidate words that occur in any candidate,
	# but the same word cannot contribute more than max_ref[word] to the toal sum.
	word_to_count = defaultdict(int)
	for word in candidate:
		# if it occurs in ANY candidate translation, then add it to the dict
		found = False
		for ref in references:
			if word in ref:
				found = True
				break
		if found:
			word_to_count[word]+=1
	# now, apply the clipping
	# iterate thorugh the keys, clipping frm max_ref_counts
	for k in word_to_count.keys():
		clip_val = max_ref_counts[k]
		word_to_count[k] = min(word_to_count[k], clip_val)
	modified_precision_score = sum(word_to_count.values())
	return modified_precision_score/len(candidate), word_to_count



if __name__ == '__main__':
	# test the modified n gram precision
	candidate = "the the the the the the the".split()
	print(candidate)
	ref1 = "the cat is on the mat".split()
	ref2 = "there is a cat on the mat".split()
	print(ref1, ref2)
	val, d = modified_n_gram_precision(candidate, [ref1, ref2])
	print(val) # should be 2
	print(d)

	candidate = "It is a guide to action which ensures that the military always obeys ethe commands of the party".split()
	r1 = "It is a guide to action that ensures that the military will forever heed Party commands".split()
	r2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party".split()
	r3 = "It is the practical guide for the army always to heed the directions of the party".split()
	val, d = modified_n_gram_precision(candidate, [r1, r2, r3])
	print(val)
	print(d)