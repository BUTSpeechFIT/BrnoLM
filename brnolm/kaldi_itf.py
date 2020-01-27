
def split_nbest_key(key):
    fields = key.split('-')
    segment = '-'.join(fields[:-1])
    trans_id = fields[-1]

    return segment, trans_id
