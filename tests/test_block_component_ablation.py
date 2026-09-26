import pytest
import torch

from test_revisiting_opd_block_policy_loss import compute_policy_loss


def objective(current, old, adv, mask, mode):
    return compute_policy_loss(
        old, current, adv, mask, cliprange=.2, cliprange_low=.2,
        cliprange_high=.2, opd_block_size=3, opd_block_advantage_mode='mean',
        opd_block_ablation=mode,
    )[0]


def manual(current, old, adv, mask, mode):
    terms = []
    for row in range(len(mask)):
        for start in range(0, mask.shape[1], 3):
            selected = mask[row, start:start+3].bool()
            if not selected.any():
                continue
            a = adv[row, start:start+3][selected].detach().mean()
            delta = (current[row, start:start+3][selected]
                     - old[row, start:start+3][selected].detach())
            ratios = delta.exp() if mode == 'adv_only' else delta.sum().exp().reshape(1)
            primary = torch.maximum(-a * ratios, -a * ratios.clamp(.8, 1.2))
            terms.extend(torch.where(a < 0, torch.minimum(-3*a, primary), primary).unbind())
    return torch.stack(terms).sum() / (mask.sum() + 1e-8)


@pytest.mark.parametrize('mode', ['adv_only', 'joint_tokenmean'])
@pytest.mark.parametrize('at_old', [False, True])
def test_component_objective_matches_hand_derivation_with_partial_mask(mode, at_old):
    old = torch.zeros((2, 5), dtype=torch.float64, requires_grad=True)
    cur = torch.tensor([[.1, -.1, .4, -.6, .9], [.7, -.3, .1, -.2, .3]],
                       dtype=torch.float64) * (0 if at_old else 1)
    cur.requires_grad_()
    adv = torch.tensor([[1., -2., 4., -5., 9.], [-2., 1., -3., 7., 8.]],
                       dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1., 1., 1., 1., 0.], [1., 0., 1., 0., 0.]])
    loss = objective(cur, old, adv, mask, mode)
    expected = manual(cur, old, adv, mask, mode)
    torch.testing.assert_close(loss, expected)
    grad, = torch.autograd.grad(loss, cur, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, cur)
    torch.testing.assert_close(grad, expected_grad)
    loss.backward()
    assert old.grad is None and adv.grad is None
    assert (cur.grad[~mask.bool()] == 0).all()


def test_b_c_gradients_match_at_old_but_historical_full_block_is_three_times():
    old = torch.zeros((1, 6))
    adv = torch.tensor([[1., 2., 3., -4., -2., -3.]])
    mask = torch.ones_like(old)
    grads = {}
    for mode in ('legacy', 'adv_only', 'joint_tokenmean'):
        cur = old.clone().requires_grad_()
        objective(cur, old, adv, mask, mode).backward()
        grads[mode] = cur.grad
    torch.testing.assert_close(grads['adv_only'], grads['joint_tokenmean'])
    torch.testing.assert_close(grads['legacy'], 3 * grads['adv_only'])


def test_joint_ratio_can_clip_when_each_token_ratio_does_not():
    old = torch.zeros((1, 3))
    adv = mask = torch.ones_like(old)
    grads = []
    for mode in ('adv_only', 'joint_tokenmean'):
        cur = torch.full_like(old, .1, requires_grad=True)
        objective(cur, old, adv, mask, mode).backward()
        grads.append(cur.grad)
    assert (grads[0] < 0).all()
    assert (grads[1] == 0).all()


@pytest.mark.parametrize('mode', ['adv_only', 'joint_tokenmean', 'token_scale'])
def test_masked_nonfinite_inputs_and_empty_rows(mode):
    old = torch.tensor([[0., float('-inf'), 0.], [float('-inf')]*3])
    cur = old.clone().requires_grad_()
    adv = torch.tensor([[1., float('nan'), -2.], [float('nan')]*3])
    mask = torch.tensor([[1., 0., 1.], [0., 0., 0.]])
    loss = objective(cur, old, adv, mask, mode)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(cur.grad).all()


def test_unknown_mode_and_incompatible_window_fail_closed():
    values = torch.zeros((1, 3))
    mask = torch.ones_like(values)
    with pytest.raises(ValueError, match='ablation'):
        objective(values, values, values, mask, 'typo')
    with pytest.raises(ValueError, match='fixed'):
        compute_policy_loss(values, values, values, mask, cliprange=.2,
            opd_block_size=3, opd_block_advantage_mode='mean',
            opd_block_ablation='adv_only', opd_window_mode='sliding')


@pytest.mark.parametrize('at_old', [False, True])
def test_scale_only_matches_token_loss_with_valid_block_counts(at_old):
    old = torch.zeros((2, 5), dtype=torch.float64, requires_grad=True)
    cur = torch.tensor([[.3, -.5, .1, .6, -.2], [-.2, .4, -.3, .2, -.1]],
                       dtype=torch.float64) * (0 if at_old else 1)
    cur.requires_grad_()
    adv = torch.tensor([[1., -2., 4., -5., 9.], [-2., 1., -3., 7., 8.]],
                       dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1., 1., 1., 1., 0.], [1., 0., 1., 1., 1.]])
    counts = torch.tensor([[3., 3., 3., 1., 1.], [2., 2., 2., 2., 2.]])
    expected = compute_policy_loss(old.detach(), cur, adv.detach() * counts, mask,
                                   cliprange=.2, opd_block_size=1)[0]
    actual = objective(cur, old, adv, mask, 'token_scale')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, cur, retain_graph=True)[0],
                               torch.autograd.grad(expected, cur, retain_graph=True)[0])
    actual.backward()
    assert old.grad is None and adv.grad is None
    assert (cur.grad[~mask.bool()] == 0).all()


def test_scale_only_does_not_broadcast_advantages():
    old = torch.zeros((1, 3))
    adv = torch.tensor([[3., -1., 1.]])
    mask = torch.ones_like(old)
    gradients = {}
    for mode in ('adv_only', 'token_scale'):
        cur = old.clone().requires_grad_()
        objective(cur, old, adv, mask, mode).backward()
        gradients[mode] = cur.grad
    assert gradients['adv_only'][0, 1] < 0
    assert gradients['token_scale'][0, 1] > 0
