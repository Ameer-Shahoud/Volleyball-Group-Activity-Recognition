def assign_not_none(*values):
    value = None
    for v in values:
        if v != None:
            value = v
            break
    return value
