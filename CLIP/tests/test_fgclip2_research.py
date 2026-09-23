"""Numerical, leakage and cache-contract checks for the research readouts."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from CLIP.fgclip2_face_impressions import (
    RegressionDataset, cross_validate_predictor, fit_score_model, predict_score_model, make_cv_splits, FaceDatasetConfig, FaceImpressionPipeline, _atomic_save_npz, anchored_cv_splits,
    center_groups,
)
from CLIP.fgclip2_legacy_research import (
    CVConfig, InnerCVSearch, evaluate_impression, development_protocol, compare_prompt_spaces, phrase_contributions, trust_center_scores, fit_trust_recipe, predict_trust_readout, trust_outer_cv, trust_tail_metrics, TrustworthinessPredictor, trust_change_metrics, fit_transfer_source, transfer_basis, fit_trust_transfer, trust_transfer_cv,
)

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(81)
        self.x = self.rng.normal(size=(72, 5)) * [1, 4, 0.02, 2, 1] + [3, 1, 2, 4, 0]
        self.y = self.x[:, 0] + 0.2 * self.x[:, 1] + self.rng.normal(size=72) * .2
        self.config = CVConfig(outer_folds=3, inner_folds=3, population=8,
                               elite=2, generations=3, bootstrap_samples=5)

    def test_ridge_model_matches_sklearn_with_collinearity_and_constant(self):
        x = np.column_stack([self.x, self.x[:, 0], np.ones(72)])
        alphas = [0.001, 1., 1000.]
        got = np.column_stack([predict_score_model(
            fit_score_model(x[:50], self.y[:50], alpha=a), x[50:]) for a in alphas])
        for j, alpha in enumerate(alphas):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
            expected = model.fit(x[:50], self.y[:50]).predict(x[50:])
            np.testing.assert_allclose(got[:, j], expected, atol=1e-9)

    def test_inner_cv_exact_mse_and_alpha(self):
        search = InnerCVSearch(self.x, self.y, self.config, 23)
        columns = (0, 2, 4)
        losses = []
        for alpha in self.config.ridge_alphas:
            predictions, _ = cross_validate_predictor(
                self.x[:, columns], self.y,
                lambda x,y,t,f: make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x,y).predict(t),
                splits=search.splits)
            losses.append(np.mean((predictions-self.y)**2))
        mse, alpha = search.ridge(columns)
        self.assertAlmostEqual(mse, min(losses), places=11)
        self.assertEqual(alpha, self.config.ridge_alphas[int(np.argmin(losses))])

    def test_float32_inputs_use_float64_search(self):
        x = self.x.astype(np.float32)
        a = InnerCVSearch(x, self.y, self.config, 23)
        b = InnerCVSearch(x.astype(np.float64), self.y, self.config, 23)
        self.assertEqual(a.x.dtype, np.dtype('float64'))
        self.assertEqual(a.ridge((0, 1, 2)), b.ridge((0, 1, 2)))
        self.assertEqual(a.lasso(), b.lasso())

    def test_oof_matches_sklearn_manual_and_rejects_bad_coverage(self):
        splits = make_cv_splits(72, 3, 12)x
        predicted, _ = cross_validate_predictor(self.x, self.y,
            lambda x,y,t,f: predict_score_model(fit_score_model(x,y,'linear'),t), splits=splits)
        for train, test in splits:
            expected = LinearRegression().fit(self.x[train], self.y[train]).predict(self.x[test])
            np.testing.assert_allclose(predicted[test], expected, atol=1e-10)
        with self.assertRaises(ValueError):
            cross_validate_predictor(self.x, self.y, None, splits=splits[:-1])
        with self.assertRaises(ValueError):
            cross_validate_predictor(self.x, self.y, None, splits=[(np.arange(72), np.arange(72))])

    def test_heldout_labels_cannot_change_own_predictions(self):
        def data(y):
            return RegressionDataset('synthetic', tuple(Path(f'{i}.jpg') for i in range(72)),
                                     tuple(f'phrase {i}' for i in range(5)), y, np.ones(72), self.x)
        a = evaluate_impression(data(self.y), self.config)
        changed = self.y.copy()
        test = np.flatnonzero(a['fold_ids'] == 0)
        changed[test] += 100
        b = evaluate_impression(data(changed), self.config)
        np.testing.assert_allclose(a['predictions'][test], b['predictions'][test], atol=0, rtol=0)
        self.assertEqual(a['fold_details'][0], b['fold_details'][0])
        for detail in a['fold_details']:
            values = [s['inner_mse'] for s in detail['evolution']]
            self.assertTrue(np.all(np.diff(values) <= 0))
            self.assertIn(0, detail['models']['evolved_ridge']['columns'])

    def test_group_splits_disjoint_at_both_levels(self):
        groups = np.repeat(np.arange(24), 3)
        for train, test in make_cv_splits(72, 3, 12, groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for a,b in make_cv_splits(len(train), 3, 13, groups[train]):
                self.assertFalse(set(groups[train][a]) & set(groups[train][b]))

    def test_portable_models_prediction_parity(self):
        from sklearn.isotonic import IsotonicRegression
        model = fit_score_model(self.x, self.y, 'isotonic', [0])
        expected = IsotonicRegression(out_of_bounds='clip').fit(self.x[:,0], self.y).predict(self.x[:,0])
        np.testing.assert_allclose(predict_score_model(model,self.x), expected)
        model = fit_score_model(self.x, self.y, 'ridge', alpha=2.)
        expected = make_pipeline(StandardScaler(), Ridge(alpha=2.)).fit(self.x,self.y).predict(self.x)
        np.testing.assert_allclose(predict_score_model(model,self.x), expected, atol=1e-10)

    def test_prefix_cache_only_encodes_new_faces(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(3): (root / f'{i}.jpg').write_bytes(bytes([i]))
            engine = SimpleNamespace(model_id='fake', revision='pinned', dtype=torch.float32)
            pipeline = FaceImpressionPipeline(engine, FaceDatasetConfig(root, cache_dir=root/'cache', limit=None))
            paths = pipeline.image_paths()
            key, manifest = pipeline._feature_identity(paths[:2], 128)
            _atomic_save_npz(root/'cache'/f'features-{key}.npz',
                             features=np.ones((2,4),np.float32), paths=np.array([p.name for p in paths[:2]]),
                             metadata=np.asarray(__import__('json').dumps(manifest)))
            with patch.object(pipeline,'_encode_features', return_value=(torch.zeros((1,4)),1)) as encode:
                features=pipeline.features()
                self.assertEqual(encode.call_args.args[0], paths[2:])
                self.assertEqual(tuple(features.features.shape),(3,4))
                self.assertTrue(pipeline.features().cache_hit)

    def test_development_faces_never_scored_in_anchored_cv(self):
        paths=tuple(Path(f'{i}.jpg') for i in range(72))
        protocol=development_protocol(paths,development_size=24)
        coverage=np.zeros(72,dtype=int)
        for train,test in anchored_cv_splits(72,protocol['development_indices'],protocol['evaluation_indices'],folds=3):
            self.assertTrue(set(protocol['development_indices'])<=set(train))
            self.assertFalse(set(train)&set(test))
            coverage[test]+=1
        np.testing.assert_array_equal(coverage[protocol['development_indices']],0)
        np.testing.assert_array_equal(coverage[protocol['evaluation_indices']],1)

    def test_expansion_heldout_labels_cannot_affect_own_fit(self):
        paths=tuple(Path(f'{i}.jpg') for i in range(72))
        def data(y):
            return RegressionDataset('synthetic',paths,tuple(str(i) for i in range(5)),y,np.ones(72),self.x)
        protocol=development_protocol(paths,development_size=24)
        spaces={'original16':[0,1,2],'similar32':[0,1,2,3],'exploratory32':[0,1,2,4]}
        a=compare_prompt_spaces(data(self.y),spaces,self.config,protocol=protocol)
        changed=self.y.copy()
        selected=np.asarray(a['evaluated_indices'])[a['fold_ids']==0]
        changed[selected]+=50
        b=compare_prompt_spaces(data(changed),spaces,self.config,protocol=protocol)
        np.testing.assert_array_equal(a['predictions'][a['fold_ids']==0],b['predictions'][b['fold_ids']==0])
        self.assertEqual(a['fold_details'][0],b['fold_details'][0])
        self.assertEqual(a['n'],48)

    def test_contribution_reference_addition_is_zero(self):
        data=RegressionDataset('synthetic',tuple(Path(f'{i}.jpg') for i in range(72)),
                               tuple(str(i) for i in range(5)),self.y,np.ones(72),self.x)
        result=phrase_contributions(data,self.config)
        self.assertEqual(result['rows'][0]['added_to_reference_delta_r2'],0.)

    def test_trust_centering_removes_per_image_per_resolution_offsets(self):
        x=self.rng.normal(size=(72,32))
        shifted=x+np.repeat(self.rng.normal(size=(72,2)),16,axis=1)
        np.testing.assert_allclose(trust_center_scores(x),trust_center_scores(shifted),atol=1e-14)
        recipe=dict(family='centered_rbf',bank='both',gamma=.1,alpha=.1)
        model=fit_trust_recipe(x[:50],self.y[:50],recipe)
        np.testing.assert_allclose(predict_trust_readout(model,x[50:]),
                                   predict_trust_readout(model,shifted[50:]),atol=1e-12)

    def test_trust_nested_selection_ignores_own_heldout_labels(self):
        x=self.rng.normal(size=(72,32))
        recipes=[dict(family='screened_ridge',bank='both',screen_k=8,alpha=a) for a in (1.,10.)]
        recipes += [dict(family='rbf',bank='p256',gamma=.1,alpha=.1),
                    dict(family='poly2',bank='both',alpha=1.)]
        protocol=development_protocol(tuple(Path(f'{i}.jpg') for i in range(72)),development_size=24)
        a=trust_outer_cv(x,self.y,recipes,protocol,folds=3,inner_folds=3)
        changed=self.y.copy();mask=a['fold_ids']==0
        changed[np.asarray(a['indices'])[mask]]+=20
        b=trust_outer_cv(x,changed,recipes,protocol,folds=3,inner_folds=3)
        np.testing.assert_array_equal(a['predictions'][mask],b['predictions'][mask])
        self.assertEqual(a['fold_details'][0],b['fold_details'][0])
        self.assertEqual(len(a['indices']),48)

    def test_trust_kernel_reference_and_serialization_parity(self):
        import joblib
        from sklearn.kernel_ridge import KernelRidge
        x=self.rng.normal(size=(72,32))
        recipe=dict(family='rbf',bank='p256',gamma=.1,alpha=.01)
        model=fit_trust_recipe(x[:50],self.y[:50],recipe)
        scale=StandardScaler().fit(x[:50,16:])
        reference=KernelRidge(kernel='rbf',gamma=.1/16,alpha=.01).fit(
            scale.transform(x[:50,16:]),self.y[:50]-self.y[:50].mean())
        expected=reference.predict(scale.transform(x[50:,16:]))+self.y[:50].mean()
        np.testing.assert_allclose(predict_trust_readout(model,x[50:]),expected,atol=1e-12)
        bundle=dict(format_version=1,target='trustworthy',phrases=list(range(16)),model=model)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(bundle,path)
            loaded=TrustworthinessPredictor.load(path)
            np.testing.assert_array_equal(loaded.predict_scores(x[50:]),predict_trust_readout(model,x[50:]))
            with self.assertRaises(ValueError):loaded.predict_scores(x[:,:4])

    def test_trust_close_pairs_use_same_fold_and_rating_gap(self):
        y=np.array([.7,.725,.745,.8]);prediction=np.array([.72,.71,.74,.81])
        result=trust_tail_metrics(y,prediction,np.array([0,0,0,1]),np.ones(4,bool),np.full(4,.001))
        self.assertEqual(result['close_pairs'],3)
        self.assertAlmostEqual(result['close_pair_concordance'],2/3)
        self.assertEqual(result['noise_separated_pairs'],3)

    def test_manipulation_metrics_remove_only_face_offset(self):
        rows=[]
        for identity,offset,slope in [('a',.12,.03),('b',-.08,-.02),('c',.05,.004)]:
            for level in range(5):
                rating=.6+slope*level
                rows.append(dict(identity=identity,method='flow',level=level,rating=rating,
                                 prediction=rating+offset,sem=.002))
        result=trust_change_metrics(rows)
        self.assertGreater(result['absolute']['rmse'],.05)
        self.assertLess(result['centered']['rmse'],1e-14)
        self.assertLess(result['baseline_changes']['rmse'],1e-14)
        self.assertLess(result['slopes']['rmse'],1e-14)
        self.assertAlmostEqual(result['centered']['pearson'],1.)
        self.assertEqual(result['direction']['0.02']['confusion'],[[1,0,0],[0,1,0],[0,0,1]])

    def test_manipulation_metrics_detect_wrong_change_direction(self):
        rows=[]
        for identity,slope in [('a',.03),('b',-.02)]:
            for level in range(5):
                rows.append(dict(identity=identity,method='vector',level=level,rating=.6+slope*level,
                                 prediction=.7-slope*level,sem=.002))
        result=trust_change_metrics(rows)
        self.assertAlmostEqual(result['centered']['pearson'],-1.)
        self.assertEqual(result['direction']['0.02']['accuracy'],0.)

    def test_transfer_contrasts_remove_identity_offsets(self):
        groups=np.repeat(np.arange(6),4)
        x=self.rng.normal(size=(24,8))
        shifted=x+np.repeat(self.rng.normal(size=(6,8)),4,axis=0)
        np.testing.assert_allclose(center_groups(x,groups),center_groups(shifted,groups),atol=1e-14)

    def test_transfer_residual_matches_augmented_least_squares(self):
        sx=self.rng.normal(size=(40,32));sy=self.rng.normal(size=40)
        vx=self.rng.normal(size=(24,32));vy=self.rng.normal(size=24);vg=np.repeat(np.arange(6),4)
        source=fit_transfer_source(sx,sy)
        basis=transfer_basis(sx,vx,source,vy,vg,'linear')
        recipe=dict(family='residual',mapping='linear',alpha=.1,vector_weight=2.,contrast=4.)
        model=fit_trust_transfer(sx,sy,vx,vy,vg,recipe,source_model=source,basis=basis)
        z=(trust_center_scores(np.row_stack([sx,vx]))-basis['mean'])/basis['scale']
        phi=np.column_stack([z,np.ones(len(z))]);a,b=phi[:40],phi[40:]
        residual=vy-predict_trust_readout(source,vx)
        design=np.row_stack([a/np.sqrt(40),np.sqrt(2/24)*b,
                             np.sqrt(8/24)*center_groups(b,vg),np.sqrt(.1)*np.eye(33)])
        target=np.concatenate([np.zeros(40),np.sqrt(2/24)*residual,
                               np.sqrt(8/24)*center_groups(residual,vg),np.zeros(33)])
        expected=np.linalg.lstsq(design,target,rcond=None)[0]
        np.testing.assert_allclose(model['coefficients'],expected,atol=1e-11)
        import joblib
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'transfer.joblib'
            joblib.dump(dict(format_version=1,target='trustworthy',phrases=list(range(16)),model=model),path)
            np.testing.assert_array_equal(TrustworthinessPredictor.load(path).predict_scores(vx),
                                          predict_trust_readout(model,vx))

    def test_transfer_validation_labels_cannot_affect_own_fold_predictions(self):
        sx=self.rng.normal(size=(48,32));sy=self.rng.normal(size=48)
        vx=self.rng.normal(size=(24,32));vy=self.rng.normal(size=24);vg=np.repeat(np.arange(6),4)
        recipes=[dict(family='source_only'),dict(family='pooled_ridge',alpha=10.,vector_weight=1.),
                 dict(family='residual',mapping='linear',alpha=.1,vector_weight=1.,contrast=4.),
                 dict(family='residual',mapping='nystrom',alpha=.01,vector_weight=1.,contrast=4.)]
        a=trust_transfer_cv(sx,sy,vx,vy,vg,recipes,folds=3)
        ss=np.array(a['folds'][0]['source_validation_indices']);vv=np.array(a['folds'][0]['vector_validation_indices'])
        new_sy=sy.copy();new_vy=vy.copy();new_sy[ss]+=20;new_vy[vv]-=20
        b=trust_transfer_cv(sx,new_sy,vx,new_vy,vg,recipes,folds=3)
        np.testing.assert_array_equal(a['source_predictions'][ss],b['source_predictions'][ss])
        np.testing.assert_array_equal(a['vector_predictions'][vv],b['vector_predictions'][vv])
        for fold in a['folds']:
            self.assertFalse(set(fold['vector_training_identities'])&set(fold['vector_validation_identities']))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
