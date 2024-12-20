from web3 import Web3

web3=Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

account_1 = "0x372b6691018cb80E931433936297b982eC64817E"
account_2 = "0x6cc2BBDd12f5224f2C301cAa763d20489954cfd4"

private_key = "0x67032632609abf14f23aa448c7c7c12c9c0ab48767256884ad0b4f4d71cde95a"

nonce = web3.eth.get_transaction_count(account_1)

tx={
    'nonce':nonce,
    'to':account_2,
    'value':web3.to_wei(1,'ether'),
    'gas': 2000000,
    'gasPrice': web3.to_wei('50','gwei')
}

signed_tx = web3.eth.account.sign_transaction(tx,private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
print(tx_hash)