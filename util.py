class Group:
    '''
    used for grouping statistics
    '''
    def __init__(self):
        self.groups = {}

    def __getitem__(self, name):
        if name not in self.groups:
            self.groups[name] = []

        return self.groups[name]

    def append(self, name, value):
        if name not in self.groups:
            self.groups[name] = []
        
        self.groups[name].append(value)
    
    def items(self):
        return self.groups.items()