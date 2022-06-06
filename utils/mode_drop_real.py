import numpy as np

class Mode_Drop():
    def __init__(self, data):
        self.data = data
        self.data_num = len(data)
        self.step = 5
        self.total = 100

    def getImage(self, images, initbox):
        imagelist = []
        for box in initbox:
            splitimages = images[box]
            imagelist.append(splitimages)
        return imagelist


    def loopprob(self, mode, initbox, state, indexes, images, image_state):
        if state == 'simultaneous':
            for i in range(self.step):
                simul = self.total - (i+1) * 20
                initbox = self.loop_simultaneous(mode, initbox, indexes)
                origin_images = self.getImage(images, initbox)
                np.savez(f'{image_state}_simul/simul_{str(simul)}.npz', image=origin_images, label=initbox)

                for box in initbox:
                    print(len(box))
                print('-'*10)
            return initbox

    def loop_simultaneous(self, mode, initbox, indexes, prob=20):
        # simultaneous mode drop
        for c, c_idx_set in enumerate(initbox):
            # 모드가 유지되고 더해지는 거라면
            if c in mode:
                complement = list(set(indexes[c]) -  set(c_idx_set))
                initbox[c] = np.append(c_idx_set, np.random.choice(complement, prob, False))
            # 모드가 사라지는 거라면
            else:
                if len(c_idx_set) < int(prob):
                    continue
                initbox[c] = list(set(c_idx_set) - set(np.random.choice(c_idx_set, prob, False)))
        return initbox

    def sequential(self, initbox, indexes):
        num = len(initbox.pop())
        div = len(initbox)
        value, left = divmod(div, num)
        # sequential mode drop
        for c, c_idx_set in enumerate(initbox):
            complement = list(set(indexes[c]) -  set(c_idx_set))
            initbox[c] = np.append(c_idx_set, np.random.choice(complement, value, False))
            if c == -1:
                complement = list(set(indexes[c]) -  set(c_idx_set))
                initbox[c] = np.append(c_idx_set,  np.random.choice(complement, left, False))
            # 모드가 사라지는 거라면
            else:
                if len(c_idx_set):
                    continue
                initbox[c] = list(set(c_idx_set) - set(np.random.choice(c_idx_set, num, False)))
        return initbox

    def loop_sequential():

