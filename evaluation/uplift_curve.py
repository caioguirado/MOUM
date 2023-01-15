class UpliftCurve:
    def __init__(self, dataset, uplift):
        self.dataset = dataset
        self.uplift = uplift
        self.combined_df = self.dataset.assign(uplift=self.uplift)

    def get_group(self, w):
        return (
            self
            .combined_df
            .query('w == @w')            
        )

    def get_group_sum(self, k, w):
        return (
            self
            .get_group(w)
            .sort_values(by=['uplift'], ascending=False)
            .head(k)
            ['Y_obs']
            .sum()
        )

    def get_uplift(self, p):
        t_size = len(self.get_group(w=1))
        c_size = len(self.get_group(w=0))

        # eq. 9 in https://arxiv.org/abs/2002.05897
        r_t = self.get_group_sum(k=p*t_size, w=1)
        r_c = self.get_group_sum(k=p*c_size, w=0)

        return r_t - r_c * (t_size / c_size)