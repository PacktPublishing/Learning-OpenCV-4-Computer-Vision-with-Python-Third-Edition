from digits_ann import create_ann, train, test


ann, test_data = train(create_ann())
test(ann, test_data)
