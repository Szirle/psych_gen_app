"""Numerical and leakage-contract tests; no encoder or GPU is required."""
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits

from ..race_perception import (
    apply_strategies, cv_splits, feature_view, fit_bundle_readout,
    predict_candidates, predict_readout, select_strategies,
)
from ..race_perception.phrases import TARGETS, build_phrase_bank


class MultiOutputTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(37)
        self.x = rng.normal(size=(32, 16))
        self.test = rng.normal(size=(9, 16))
        self.y = rng.normal(size=(32, 8)) + np.arange(8)[None, :]
        self.limit = threadpool_limits(limits=1)

    def tearDown(self):
        self.limit.restore_original_limits()

    def test_linear_path_matches_sklearn_per_output(self):
        recipes = [dict(family="linear", bank="both", centered=False, alpha=a) for a in (.001, .1, 10.)]
        result = predict_candidates(self.x, self.y, self.test, recipes)
        mean, scale = self.x.mean(axis=0), self.x.std(axis=0)
        for j, recipe in enumerate(recipes):
            model = Ridge(alpha=recipe["alpha"]*self.x.shape[1]).fit((self.x-mean)/scale, self.y)
            expected = model.predict((self.test-mean)/scale)
            np.testing.assert_allclose(result[:,j], expected, atol=1e-10, rtol=1e-10)

    def test_nonlinear_path_matches_independent_centered_kernel(self):
        from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
        recipes = [dict(family="rbf", bank="both", centered=True, gamma=.3, alpha=.01),
                   dict(family="poly2", bank="p256", centered=False, alpha=.1)]
        actual = predict_candidates(self.x,self.y,self.test,recipes)
        for j, recipe in enumerate(recipes):
            a,b=feature_view(self.x,recipe),feature_view(self.test,recipe)
            mean,scale=a.mean(axis=0),a.std(axis=0)
            a,b=(a-mean)/scale,(b-mean)/scale
            if recipe["family"]=="rbf":
                k=rbf_kernel(a,gamma=.3/a.shape[1]);cross=rbf_kernel(b,a,gamma=.3/a.shape[1])
            else:
                k=polynomial_kernel(a,degree=2,gamma=1/a.shape[1],coef0=1)
                cross=polynomial_kernel(b,a,degree=2,gamma=1/a.shape[1],coef0=1)
            km=k.mean(axis=0);kg=k.mean()
            kc=k-km[None,:]-km[:,None]+kg
            cc=cross-cross.mean(axis=1,keepdims=True)-km[None,:]+kg
            fit=KernelRidge(alpha=recipe["alpha"],kernel="precomputed").fit(kc,self.y-self.y.mean(axis=0))
            expected=fit.predict(cc)+self.y.mean(axis=0)
            np.testing.assert_allclose(actual[:,j],expected,atol=1e-10,rtol=1e-10)

    def test_bundle_blends_preserve_target_and_phrase_order(self):
        recipes=[dict(family="linear",bank="both",centered=False,alpha=.1),
                 dict(family="rbf",bank="both",centered=True,gamma=.1,alpha=.03)]
        strategies=[dict(indices=[0,1],weights=[t/7,1-t/7]) for t in range(8)]
        model=fit_bundle_readout(self.x,self.y,recipes,strategies)
        expected=apply_strategies(predict_candidates(self.x,self.y,self.test,recipes),strategies)
        np.testing.assert_allclose(predict_readout(model,self.test),expected,atol=1e-10)
        with self.assertRaises(ValueError):
            predict_readout(model,self.test[:,:14])
        bad=self.test.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):
            predict_readout(model,bad)

    def test_groups_disjoint_and_complete_at_both_levels(self):
        groups=np.repeat(np.arange(12),3)
        outer=cv_splits(len(groups),6,13,groups)
        coverage=np.zeros(len(groups),int)
        for train,test in outer:
            coverage[test]+=1
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for a,b in cv_splits(len(train),5,14,groups[train]):
                self.assertFalse(set(groups[train[a]]) & set(groups[train[b]]))
                self.assertFalse(set(train[a]) & set(test))
        np.testing.assert_array_equal(coverage,1)

    def test_output_selection_is_separate(self):
        p=np.stack([self.y+np.arange(8)[None,:]/10,self.y+(7-np.arange(8))[None,:]/10],axis=1)
        recipes=[dict(family="linear",bank="both",centered=False,alpha=a) for a in (.1,1.)]
        strategies,ridge,losses=select_strategies(p,self.y,recipes)
        self.assertEqual(strategies[0]["indices"],[0])
        self.assertEqual(strategies[7]["indices"],[1])
        self.assertEqual(ridge[0],0);self.assertEqual(ridge[7],1)

    def test_phrase_bank_contract(self):
        bank=build_phrase_bank()
        self.assertEqual(len(bank["phrases"]),len(set(bank["phrases"])))
        self.assertEqual(tuple(bank["targets"]),TARGETS)
        self.assertEqual(len(bank["reference_indices"]),8)
        self.assertGreater(len(bank["phrases"]),500)
        self.assertEqual(set(j for ids in bank["groups"].values() for j in ids),set(range(len(bank["phrases"]))))


if __name__ == "__main__":
    unittest.main()
