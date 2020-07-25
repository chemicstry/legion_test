extern crate proc_macro;
use proc_macro::TokenStream;
// use syn::{parse_macro_input, DeriveInput};

#[proc_macro]
pub fn query(_input: TokenStream) -> TokenStream {
    //let input = syn::parse_macro_input!(input as syn::Expr);
    "fn answer() -> u32 { 42 }".parse().unwrap()
}
