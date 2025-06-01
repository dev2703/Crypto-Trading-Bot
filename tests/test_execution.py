import pytest
from strategy.execution import MarketOrder, LimitOrder, StopOrder, compute_transaction_cost, compute_slippage, execute_order

def test_market_order():
    order = MarketOrder('BTC', 1.0)
    assert order.symbol == 'BTC'
    assert order.quantity == 1.0
    assert order.order_type == 'market'

def test_limit_order():
    order = LimitOrder('ETH', 10.0, 2000.0)
    assert order.symbol == 'ETH'
    assert order.quantity == 10.0
    assert order.order_type == 'limit'
    assert order.limit_price == 2000.0

def test_stop_order():
    order = StopOrder('XRP', 100.0, 0.5)
    assert order.symbol == 'XRP'
    assert order.quantity == 100.0
    assert order.order_type == 'stop'
    assert order.stop_price == 0.5

def test_compute_transaction_cost():
    order = MarketOrder('BTC', 1.0)
    cost = compute_transaction_cost(order, 50000.0, 0.001)
    assert cost == 50.0

def test_compute_slippage():
    order = MarketOrder('BTC', 1.0)
    slippage = compute_slippage(order, 50000.0, 0.001)
    assert slippage == 50.0

def test_execute_order():
    order = MarketOrder('BTC', 1.0)
    result = execute_order(order, 50000.0, 0.001, 0.001)
    assert result['order'] == order
    assert result['market_price'] == 50000.0
    assert result['transaction_cost'] == 50.0
    assert result['slippage'] == 50.0
    assert result['total_cost'] == 100.0 